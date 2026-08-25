// Chapter 15: Advanced optimizations for matrix multiplication
// §15.8  Software pipelining -- double-buffered tile prefetch, Fig. 15.14.
// §15.9  Specialized software and hardware support -- cuda::memcpy_async /
// cp.async, mentioned as the hardware-assisted alternative to
// compiler-scheduled instruction interleaving.
//
// Builds on 04_matmul_bank_conflict_free.cu (§15.3-§15.6): identical
// block-level tiling, register-tiled mm(), warp/quadrant/lane-based
// coalesced writeTile(), and BK+1-padded A_s. The tile loop itself is
// restructured per §15.8.
//
// The problem (§15.7-§15.8): with a single pair of shared-memory buffers,
// each thread block alternates between a memory-bound phase (loadTile,
// FP ALUs idle) and a compute-bound phase (mm, memory hardware idle),
// separated by two __syncthreads() barriers -- one enforcing a true
// dependence (must finish loading before computing) and one enforcing a
// false dependence (must finish computing with the old tile before
// overwriting it with the new one, purely because they share one buffer).
// §15.8 (Fig. 15.14) removes the false dependence with double buffering
// (Acurr_s/Bcurr_s for compute, Anext_s/Bnext_s for the concurrently
// in-flight next load), so the next tile's load can be issued and start
// executing while the current tile's compute is still in flight, without
// any barrier between them.
//
// This file combines that double-buffering structure with §15.9's
// cuda::memcpy_async (which the CUDA C++ standard library lowers to the
// hardware cp.async / LDGSTS instruction on SM80+), instead of the book's
// plain (implicitly synchronous) loadTile: each thread issues its tile-load
// elements as an asynchronous copy from global memory directly into shared
// memory, tracked by a thread-scoped cuda::pipeline, so the copy genuinely
// executes in the background while this thread's own FMA instructions for
// the current tile run -- true producer/consumer overlap between the
// async-copy engine and the compute pipeline, not just the removal of a
// false dependence for the compiler to (maybe) exploit. This mirrors this
// repo's ch06/06_double_buffering_async_copy.cu, which applies the same
// combination to Chapter 5's plain tiled matmul; here it is layered on top
// of every optimization from this chapter instead.
//
// cuda::memcpy_async targeting shared memory requires SM80 (Ampere) or
// newer, so this file is compiled with -arch=sm_80 (per this chapter's
// Makefile) and, like ch06/06, checks the actual device's compute
// capability at run time and exits gracefully below 8.0. Because
// cuda::memcpy_async copies a fixed, unconditional byte span (no per-element
// bounds check like loadTile's), this file -- like ch06/06 -- requires M,
// N, and K to be exact multiples of the tile dimensions (BM, BN, BK); it
// does not attempt the boundary-checked case.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cooperative_groups.h>
#include <cuda/pipeline>

#include "../../common/cuda_utils.h"

#define BM 128
#define BN 128
#define BK 8
#define BK_PADDED (BK + 1)  // §15.6: bank-conflict-free padding, carried over from 04

#define NUM_THREADS_PER_BLOCK 256

#define WARPS_PER_BLOCK_Y 2
#define WARPS_PER_BLOCK_X 4
#define WARP_TILE_M (BM / WARPS_PER_BLOCK_Y)  // 64
#define WARP_TILE_N (BN / WARPS_PER_BLOCK_X)  // 32
#define QUAD_M (WARP_TILE_M / 2)  // 32
#define QUAD_N (WARP_TILE_N / 2)  // 16
#define LANES_PER_WARP_X 4
#define QM 4
#define QN 4
#define NUM_QUADS 4

// ---------------------------------------------------------------------------
// §15.9: async counterpart of Fig. 15.5's loadTile(). Same sub-tile
// decomposition (one element per thread per sub-tile), but each element is
// issued as a cuda::memcpy_async copy instead of a direct assignment, and
// there is no bounds check -- callers must guarantee T's height x width
// region is entirely in-bounds (true here because M, N, K are all required
// to be exact multiples of BM, BN, BK in this file).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void loadTileAsync(const float *T, unsigned int lda, float *T_s,
                                                unsigned int ldas, unsigned int height,
                                                unsigned int width,
                                                cuda::pipeline<cuda::thread_scope_thread> &pipe) {
    unsigned int rowsPerSubTile = NUM_THREADS_PER_BLOCK / width;
    unsigned int numSubTiles = height / rowsPerSubTile;
    #pragma unroll
    for (unsigned int subTile = 0; subTile < numSubTiles; ++subTile) {
        unsigned int row = subTile * rowsPerSubTile + threadIdx.x / width;
        unsigned int col = threadIdx.x % width;
        cuda::memcpy_async(&T_s[row * ldas + col], &T[row * lda + col], sizeof(float), pipe);
    }
}

// Fig. 15.9's register-tiled mm() -- unchanged from 03/04.
__device__ __forceinline__ void mm(unsigned int m, unsigned int n, unsigned int k, const float *a,
                                    unsigned int lda, const float *b, unsigned int ldb,
                                    float c[][QN]) {
    #pragma unroll
    for (unsigned int i = 0; i < k; ++i) {
        float a_r[QM];
        #pragma unroll
        for (unsigned int row = 0; row < m; ++row) {
            a_r[row] = a[row * lda + i];
        }

        float b_r[QN];
        #pragma unroll
        for (unsigned int col = 0; col < n; ++col) {
            b_r[col] = b[i * ldb + col];
        }

        #pragma unroll
        for (unsigned int row = 0; row < m; ++row) {
            #pragma unroll
            for (unsigned int col = 0; col < n; ++col) {
                c[row][col] += a_r[row] * b_r[col];
            }
        }
    }
}

// §15.5's coalesced writeTile() -- unchanged from 03/04. M and N are exact
// multiples of BM/BN in this file, so every quadrant sub-tile is fully
// in-bounds and the float4 fast path is always taken.
__device__ __forceinline__ void writeTileCoalesced(float *C, unsigned int N,
                                                     unsigned int rowBase[NUM_QUADS],
                                                     unsigned int colBase[NUM_QUADS],
                                                     float C_r[NUM_QUADS][QM][QN]) {
    #pragma unroll
    for (unsigned int q = 0; q < NUM_QUADS; ++q) {
        #pragma unroll
        for (unsigned int r = 0; r < QM; ++r) {
            unsigned int row = rowBase[q] + r;
            float *dst = &C[row * N + colBase[q]];
            float4 v = make_float4(C_r[q][r][0], C_r[q][r][1], C_r[q][r][2], C_r[q][r][3]);
            *reinterpret_cast<float4 *>(dst) = v;
        }
    }
}

// ---------------------------------------------------------------------------
// §15.8, Fig. 15.14 + §15.9: double-buffered, software-pipelined main
// kernel. Two physical A_s/B_s buffers (index 0 and 1) ping-pong across
// tiles. Each thread owns a thread-scoped cuda::pipeline tracking its own
// outstanding memcpy_async copies:
//   - tile 0's load is issued before the loop starts (priming).
//   - at the top of iteration `tile`, tile+1's load is issued into the
//     OTHER buffer *before* this thread waits for tile's own data -- so the
//     copy engine has this whole iteration's compute to finish delivering
//     the next tile (the double-buffering half of Fig. 15.14: no barrier
//     needed between "read curr" and "write next" since they're different
//     buffers).
//   - consumer_wait() blocks only until THIS thread's own oldest
//     outstanding copy lands; __syncthreads() is still required afterward
//     so every thread's tile elements (loaded by other threads) are visible
//     before any thread reads the whole shared tile.
// ---------------------------------------------------------------------------
__global__ void mm_tiled_pipelined_kernel(const float *__restrict__ A,
                                           const float *__restrict__ B, float *__restrict__ C,
                                           unsigned int M, unsigned int N, unsigned int K) {
    unsigned int bRow = blockIdx.y * BM;
    unsigned int bCol = blockIdx.x * BN;

    unsigned int warpId = threadIdx.x / 32;
    unsigned int laneId = threadIdx.x % 32;
    unsigned int warpRow = warpId / WARPS_PER_BLOCK_X;
    unsigned int warpCol = warpId % WARPS_PER_BLOCK_X;
    unsigned int laneRow = laneId / LANES_PER_WARP_X;
    unsigned int laneCol = laneId % LANES_PER_WARP_X;

    unsigned int wRow = warpRow * WARP_TILE_M;
    unsigned int wCol = warpCol * WARP_TILE_N;

    unsigned int rowBaseLocal[NUM_QUADS];
    unsigned int colBaseLocal[NUM_QUADS];
    #pragma unroll
    for (unsigned int qr = 0; qr < 2; ++qr) {
        #pragma unroll
        for (unsigned int qc = 0; qc < 2; ++qc) {
            unsigned int q = qr * 2 + qc;
            rowBaseLocal[q] = wRow + qr * QUAD_M + laneRow * QM;
            colBaseLocal[q] = wCol + qc * QUAD_N + laneCol * QN;
        }
    }

    float C_r[NUM_QUADS][QM][QN];
    #pragma unroll
    for (unsigned int q = 0; q < NUM_QUADS; ++q) {
        #pragma unroll
        for (unsigned int r = 0; r < QM; ++r) {
            #pragma unroll
            for (unsigned int c = 0; c < QN; ++c) {
                C_r[q][r][c] = 0.0f;
            }
        }
    }

    __shared__ float A_s[2][BM * BK_PADDED];
    __shared__ float B_s[2][BK * BN];

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    auto issueLoad = [&](unsigned int tileIdx, unsigned int buf) {
        pipe.producer_acquire();
        loadTileAsync(&A[bRow * K + tileIdx * BK], K, &A_s[buf][0], BK_PADDED, BM, BK, pipe);
        loadTileAsync(&B[tileIdx * BK * N + bCol], N, &B_s[buf][0], BN, BK, BN, pipe);
        pipe.producer_commit();
    };

    // Pre-fetch first iteration tiles to shared memory (Fig. 15.14 lines 01-03).
    issueLoad(0, 0);

    unsigned int numTiles = K / BK;
    for (unsigned int tile = 0; tile < numTiles; ++tile) {
        unsigned int buf = tile % 2;

        // Pre-fetch next iteration tiles to shared memory (Fig. 15.14 lines
        // 13-15), into the OTHER buffer, issued before this thread waits
        // on / computes with the current tile.
        if (tile + 1 < numTiles) {
            issueLoad(tile + 1, 1 - buf);
        }

        // Wait for this thread's own current-tile copy to land, then a
        // block-wide barrier (Fig. 15.14 line 05 / line 17).
        pipe.consumer_wait();
        __syncthreads();

        // Compute with current iteration shared memory tiles (Fig. 15.14 line 11).
        #pragma unroll
        for (unsigned int q = 0; q < NUM_QUADS; ++q) {
            mm(QM, QN, BK, &A_s[buf][rowBaseLocal[q] * BK_PADDED], BK_PADDED,
               &B_s[buf][colBaseLocal[q]], BN, C_r[q]);
        }

        __syncthreads();
        pipe.consumer_release();
    }

    unsigned int rowBaseGlobal[NUM_QUADS];
    unsigned int colBaseGlobal[NUM_QUADS];
    #pragma unroll
    for (unsigned int q = 0; q < NUM_QUADS; ++q) {
        rowBaseGlobal[q] = bRow + rowBaseLocal[q];
        colBaseGlobal[q] = bCol + colBaseLocal[q];
    }
    writeTileCoalesced(C, N, rowBaseGlobal, colBaseGlobal, C_r);
}

void matrixMul_h(const float *A, const float *B, float *C, unsigned int M, unsigned int N,
                  unsigned int K) {
    for (unsigned int row = 0; row < M; ++row) {
        for (unsigned int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (unsigned int i = 0; i < K; ++i) {
                sum += A[row * K + i] * B[i * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

int main() {
    // §15.9 note: cuda::memcpy_async targeting shared memory (cp.async)
    // needs SM80 (Ampere) or newer hardware. Check the device we actually
    // landed on and bail out cleanly rather than crash if it doesn't
    // qualify, matching this repo's ch06/06_double_buffering_async_copy.cu.
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("Device %d: %s (compute capability %d.%d)\n", device, prop.name, prop.major, prop.minor);
    if (prop.major * 10 + prop.minor < 80) {
        printf("This sample requires compute capability >= 8.0 (Ampere or newer) for\n"
               "cuda::memcpy_async / cp.async hardware asynchronous copy support.\n"
               "Detected compute capability %d.%d on device %d (%s) -- skipping.\n",
               prop.major, prop.minor, device, prop.name);
        return 0;
    }

    // Exact multiples of BM/BN/BK -- required, since loadTileAsync has no
    // per-element bounds check (see file header).
    const unsigned int M = 1024, N = 1024, K = 1024;
    if (M % BM != 0 || N % BN != 0 || K % BK != 0) {
        fprintf(stderr, "M, N, K must be exact multiples of BM, BN, BK for this file\n");
        return 1;
    }

    size_t countA = static_cast<size_t>(M) * K;
    size_t countB = static_cast<size_t>(K) * N;
    size_t countC = static_cast<size_t>(M) * N;
    std::vector<float> A_h(countA), B_h(countB), C_ref(countC), C_h(countC);

    for (size_t i = 0; i < countA; ++i) A_h[i] = static_cast<float>(i % 13) * 0.1f - 0.6f;
    for (size_t i = 0; i < countB; ++i) B_h[i] = static_cast<float>(i % 7) * 0.2f - 0.6f;

    printf("Computing CPU reference (M=%u N=%u K=%u, %zu output elements)...\n", M, N, K, countC);
    matrixMul_h(A_h.data(), B_h.data(), C_ref.data(), M, N, K);

    size_t sizeA = countA * sizeof(float);
    size_t sizeB = countB * sizeof(float);
    size_t sizeC = countC * sizeof(float);

    float *A_d, *B_d, *C_d;
    CUDA_CHECK(cudaMalloc((void **)&A_d, sizeA));
    CUDA_CHECK(cudaMalloc((void **)&B_d, sizeB));
    CUDA_CHECK(cudaMalloc((void **)&C_d, sizeC));
    CUDA_CHECK(cudaMemcpy(A_d, A_h.data(), sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h.data(), sizeB, cudaMemcpyHostToDevice));

    dim3 dimBlock(NUM_THREADS_PER_BLOCK, 1, 1);
    dim3 dimGrid(N / BN, M / BM, 1);

    // Warm-up launch (discarded).
    mm_tiled_pipelined_kernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    mm_tiled_pipelined_kernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    float ms = timer.stopAndGetMs();
    CUDA_CHECK(cudaMemcpy(C_h.data(), C_d, sizeC, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));

    bool ok = true;
    for (size_t i = 0; i < countC; ++i) {
        if (!nearlyEqual(C_h[i], C_ref[i])) {
            ok = false;
            fprintf(stderr, "Mismatch at i=%zu: gpu=%f cpu=%f\n", i, C_h[i], C_ref[i]);
            break;
        }
    }

    printf("M=%u N=%u K=%u (BM=%d BN=%d BK=%d, A_s lda=%d, double-buffered async-copy)\n", M, N, K,
           BM, BN, BK, BK_PADDED);
    printf("Software-pipelined (§15.8/§15.9) kernel time: %.3f ms  [%s]\n", ms,
           ok ? "match" : "MISMATCH");
    printf("%s\n", ok ? "PASS" : "FAIL");

    return ok ? 0 : 1;
}
