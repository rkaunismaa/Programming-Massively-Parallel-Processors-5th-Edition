// Chapter 15: Advanced optimizations for matrix multiplication
// §15.5  Coalesced storing of the output tile -- rearranged thread-level
// output tiles, Fig. 15.10.
//
// Builds on 02_matmul_register_tiled.cu (§15.4): same block-level tiling,
// loadTile()/mm() (register-tiled)/clear(), but the thread-to-output-tile
// assignment and writeTile() are both rewritten per §15.5.
//
// The problem (§15.5): with a plain 8x8 thread-level output tile laid out
// contiguously (as in 01/02), threads in the same warp write to C locations
// 8 elements (32 B) apart -- not coalesced. The book's fix, illustrated in
// Fig. 15.10 for a 128x128 block tile with 256 threads (8 warps):
//   - the 8 warps are arranged 2x4, each warp owning a 64x32 warp-level
//     output tile;
//   - each warp-level tile is split into four 32x16 quadrants;
//   - the warp's 32 threads are arranged 8x4, so each thread takes a 4x4
//     sub-tile from EACH quadrant -- four 4x4 physical sub-tiles per
//     thread, which together are the same logical 8x8 output tile as
//     before, just non-contiguously placed.
// Because each thread's four sub-tiles are exactly 4x4, a whole row of a
// sub-tile (4 floats = 16 B) fits in one float4 vector store, and the 4
// threads sharing a quadrant row (varying lane-column, fixed lane-row) issue
// vector stores to 16 B chunks that are adjacent to each other -- coalesced,
// unlike the 8-elements-apart stores of a naively-assigned 8x8 tile.
//
// mm()'s register-tiled strip logic from Fig. 15.9 is reused unchanged, but
// is now invoked once per quadrant (m=n=QM=QN=4) against that quadrant's
// own row/column offset into the shared-memory input tiles, exactly as the
// book directs: "the mm() and writeTile() device functions need to be
// revised to iterate through these 4x4 sub-tiles of the thread-level output
// tile ... in each step" (§15.5).

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

#define BM 128
#define BN 128
#define BK 8

#define NUM_THREADS_PER_BLOCK 256

// Warp/quadrant/lane geometry for the 128x128 block tile with 256 threads
// (8 warps), per §15.5's running example.
#define WARPS_PER_BLOCK_Y 2  // warp rows
#define WARPS_PER_BLOCK_X 4  // warp cols
#define WARP_TILE_M (BM / WARPS_PER_BLOCK_Y)  // 64
#define WARP_TILE_N (BN / WARPS_PER_BLOCK_X)  // 32
#define QUAD_M (WARP_TILE_M / 2)  // 32 (quadrant height)
#define QUAD_N (WARP_TILE_N / 2)  // 16 (quadrant width)
#define LANES_PER_WARP_Y 8
#define LANES_PER_WARP_X 4
#define QM 4  // per-thread, per-quadrant sub-tile height
#define QN 4  // per-thread, per-quadrant sub-tile width
#define NUM_QUADS 4

// Fig. 15.5: loadTile() -- unchanged from 01/02.
__device__ __forceinline__ void loadTile(const float *T, unsigned int lda, unsigned int maxRow,
                                          unsigned int maxCol, float *T_s, unsigned int ldas,
                                          unsigned int height, unsigned int width) {
    unsigned int rowsPerSubTile = NUM_THREADS_PER_BLOCK / width;
    unsigned int numSubTiles = height / rowsPerSubTile;
    #pragma unroll
    for (unsigned int subTile = 0; subTile < numSubTiles; ++subTile) {
        unsigned int row = subTile * rowsPerSubTile + threadIdx.x / width;
        unsigned int col = threadIdx.x % width;
        if (row < maxRow && col < maxCol) {
            T_s[row * ldas + col] = T[row * lda + col];
        } else {
            T_s[row * ldas + col] = 0.0f;
        }
    }
}

// Fig. 15.9's register-tiled mm(), parameterized on a QMxQN (here 4x4)
// output sub-tile rather than the full 8x8 logical tile -- called once per
// quadrant below.
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

// ---------------------------------------------------------------------------
// §15.5, Fig. 15.10: coalesced writeTile(). Each thread stores its four
// QMxQN (4x4) physical sub-tiles. A whole sub-tile row (QN=4 floats) is
// stored with one float4 vector instruction when it is fully in-bounds and
// 16 B aligned (guaranteed here since bCol/quadrant/lane offsets are all
// multiples of 4 floats); otherwise it falls back to a per-element
// boundary-checked scalar store so out-of-range M/N still work correctly.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void writeTileCoalesced(float *C, unsigned int N, unsigned int M,
                                                     unsigned int rowBase[NUM_QUADS],
                                                     unsigned int colBase[NUM_QUADS],
                                                     float C_r[NUM_QUADS][QM][QN]) {
    #pragma unroll
    for (unsigned int q = 0; q < NUM_QUADS; ++q) {
        #pragma unroll
        for (unsigned int r = 0; r < QM; ++r) {
            unsigned int row = rowBase[q] + r;
            if (row >= M) continue;
            float *dst = &C[row * N + colBase[q]];
            // The float4 vector store additionally requires the destination
            // address to be 16 B aligned, i.e. (row*N + colBase[q]) a
            // multiple of 4 floats -- guaranteed when N itself is a
            // multiple of 4 (colBase[q] always is, by construction), but
            // checked explicitly here so this function stays correct for
            // any N.
            bool aligned = ((row * N + colBase[q]) % 4 == 0) && (colBase[q] + QN <= N);
            if (aligned) {
                float4 v = make_float4(C_r[q][r][0], C_r[q][r][1], C_r[q][r][2], C_r[q][r][3]);
                *reinterpret_cast<float4 *>(dst) = v;
            } else {
                #pragma unroll
                for (unsigned int c = 0; c < QN; ++c) {
                    if (colBase[q] + c < N) dst[c] = C_r[q][r][c];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Main kernel: same block-level tiling loop as Fig. 15.3, but the
// thread-level output assignment follows §15.5's warp/quadrant/lane
// decomposition instead of a single contiguous tMxtN tile, and accumulation
// + storing are done per-quadrant.
// ---------------------------------------------------------------------------
__global__ void mm_tiled_coalesced_kernel(const float *A, const float *B, float *C,
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

    // Block-tile-relative row/col base of each of this thread's 4 quadrant
    // sub-tiles (q = qr*2 + qc).
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

    for (unsigned int tile = 0; tile < (K + BK - 1) / BK; ++tile) {
        __shared__ float A_s[BM * BK];
        __shared__ float B_s[BK * BN];
        loadTile(&A[bRow * K + tile * BK], K, M - bRow, K - tile * BK, &A_s[0], BK, BM, BK);
        loadTile(&B[tile * BK * N + bCol], N, K - tile * BK, N - bCol, &B_s[0], BN, BK, BN);
        __syncthreads();

        #pragma unroll
        for (unsigned int q = 0; q < NUM_QUADS; ++q) {
            mm(QM, QN, BK, &A_s[rowBaseLocal[q] * BK], BK, &B_s[colBaseLocal[q]], BN, C_r[q]);
        }
        __syncthreads();
    }

    unsigned int rowBaseGlobal[NUM_QUADS];
    unsigned int colBaseGlobal[NUM_QUADS];
    #pragma unroll
    for (unsigned int q = 0; q < NUM_QUADS; ++q) {
        rowBaseGlobal[q] = bRow + rowBaseLocal[q];
        colBaseGlobal[q] = bCol + colBaseLocal[q];
    }
    writeTileCoalesced(C, N, M, rowBaseGlobal, colBaseGlobal, C_r);
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

float runTiled(const float *A_h, const float *B_h, float *C_h, unsigned int M, unsigned int N,
               unsigned int K) {
    size_t sizeA = static_cast<size_t>(M) * K * sizeof(float);
    size_t sizeB = static_cast<size_t>(K) * N * sizeof(float);
    size_t sizeC = static_cast<size_t>(M) * N * sizeof(float);

    float *A_d, *B_d, *C_d;
    CUDA_CHECK(cudaMalloc((void **)&A_d, sizeA));
    CUDA_CHECK(cudaMalloc((void **)&B_d, sizeB));
    CUDA_CHECK(cudaMalloc((void **)&C_d, sizeC));
    CUDA_CHECK(cudaMemcpy(A_d, A_h, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, sizeB, cudaMemcpyHostToDevice));

    dim3 dimBlock(NUM_THREADS_PER_BLOCK, 1, 1);
    dim3 dimGrid((N + BN - 1) / BN, (M + BM - 1) / BM, 1);

    mm_tiled_coalesced_kernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    mm_tiled_coalesced_kernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    float ms = timer.stopAndGetMs();

    CUDA_CHECK(cudaMemcpy(C_h, C_d, sizeC, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));

    return ms;
}

bool runTestCase(unsigned int M, unsigned int N, unsigned int K) {
    size_t countA = static_cast<size_t>(M) * K;
    size_t countB = static_cast<size_t>(K) * N;
    size_t countC = static_cast<size_t>(M) * N;
    std::vector<float> A_h(countA), B_h(countB), C_ref(countC), C_h(countC);

    for (size_t i = 0; i < countA; ++i) A_h[i] = static_cast<float>(i % 13) * 0.1f - 0.6f;
    for (size_t i = 0; i < countB; ++i) B_h[i] = static_cast<float>(i % 7) * 0.2f - 0.6f;

    matrixMul_h(A_h.data(), B_h.data(), C_ref.data(), M, N, K);
    float ms = runTiled(A_h.data(), B_h.data(), C_h.data(), M, N, K);

    bool ok = true;
    for (size_t i = 0; i < countC; ++i) {
        if (!nearlyEqual(C_h[i], C_ref[i])) {
            ok = false;
            fprintf(stderr, "Mismatch at M=%u N=%u K=%u i=%zu: gpu=%f cpu=%f\n", M, N, K, i,
                    C_h[i], C_ref[i]);
            break;
        }
    }

    printf("M=%-5u N=%-5u K=%-5u (BM=%d BN=%d BK=%d, quadrant=%dx%d): %8.3f ms  [%s]\n", M, N, K,
           BM, BN, BK, QM, QN, ms, ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    bool ok = true;
    ok = runTestCase(200, 180, 90) && ok;
    ok = runTestCase(1024, 1024, 1024) && ok;
    // N not a multiple of 4: exercises writeTileCoalesced's scalar fallback
    // for the trailing, non-16B-aligned columns of the float4 fast path.
    ok = runTestCase(131, 173, 67) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
