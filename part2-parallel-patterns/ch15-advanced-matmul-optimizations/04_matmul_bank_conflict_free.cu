// Chapter 15: Advanced optimizations for matrix multiplication
// §15.6  Eliminating bank conflicts -- padded A_s, Fig. 15.11.
//
// Builds on 03_matmul_coalesced_output_store.cu (§15.5): identical
// block-level tiling, loadTile(), register-tiled mm(), and warp/quadrant/
// lane-based coalesced writeTile(). The only change is the shared-memory
// layout of A_s.
//
// The problem (§15.6): with the warp organized 8x4 (§15.5), the first 4
// threads of a warp (lane-col 0-3, lane-row 0) load from the SAME row of
// the block-level A_s tile (since a's thread-level input tile only depends
// on lane-row, per mm()'s &A_s[rowBase*lda] offset), while lane-row 1's 4
// threads load from a row 4 below, lane-row 2's from 8 below, etc. With
// A_s's leading dimension equal to bK (=8), row r's first element sits at
// linear index r*8; for r = 0, 4, 8, ..., 28 (the 8 lane-row groups) all of
// these indices are multiples of 32, i.e. all land in shared-memory bank 0
// -- an 8-way bank conflict on every strip load (the book walks through
// this exact arithmetic for bK=8, 32 banks).
//
// The fix (§15.6, Fig. 15.11(b)): pad A_s with one extra column so its
// leading dimension is bK+1 (=9) instead of bK. Row r then starts at linear
// index r*9: for r = 0, 4, 8, ..., 28 that's 0, 36, 72, ..., 252, which mod
// 32 gives 0, 4, 8, ..., 28 -- 8 distinct banks, no conflict. B_s is left
// unchanged: the book notes each warp's B_s strip load is 32 consecutive
// elements (one per lane-col group's worth summed over the warp), already
// spread over all 32 banks with no padding needed.
//
// The book states the change explicitly: replace
//   __shared__ float A_s[bM*bK];
// with
//   __shared__ float A_s[bM*(bK+1)];
// and pass bK+1 as A_s's leading dimension to loadTile() and mm() wherever
// bK was previously passed for A_s.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

#define BM 128
#define BN 128
#define BK 8
#define BK_PADDED (BK + 1)  // §15.6, Fig. 15.11(b): +1-column padding on A_s

#define NUM_THREADS_PER_BLOCK 256

#define WARPS_PER_BLOCK_Y 2
#define WARPS_PER_BLOCK_X 4
#define WARP_TILE_M (BM / WARPS_PER_BLOCK_Y)  // 64
#define WARP_TILE_N (BN / WARPS_PER_BLOCK_X)  // 32
#define QUAD_M (WARP_TILE_M / 2)  // 32
#define QUAD_N (WARP_TILE_N / 2)  // 16
#define LANES_PER_WARP_Y 8
#define LANES_PER_WARP_X 4
#define QM 4
#define QN 4
#define NUM_QUADS 4

// Fig. 15.5: loadTile() -- unchanged logic; called with ldas = BK_PADDED
// for A_s and ldas = BN (unchanged) for B_s.
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

// Fig. 15.9's register-tiled mm() -- unchanged logic; the caller passes
// lda = BK_PADDED for the A-side thread-level input tile.
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

// §15.5's coalesced writeTile() -- unchanged from 03.
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
// Main kernel: identical to 03's mm_tiled_coalesced_kernel except A_s is
// declared BM*BK_PADDED and every load/read of A_s uses BK_PADDED as its
// leading dimension (§15.6). B_s and its leading dimension (BN) are
// unchanged, per the book's note that B_s's strip load has no conflicts.
// ---------------------------------------------------------------------------
__global__ void mm_tiled_bank_conflict_free_kernel(const float *A, const float *B, float *C,
                                                     unsigned int M, unsigned int N,
                                                     unsigned int K) {
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

    for (unsigned int tile = 0; tile < (K + BK - 1) / BK; ++tile) {
        // §15.6, Fig. 15.11(b): A_s padded to BM*(BK+1) -- BK_PADDED is its
        // leading dimension everywhere it's loaded from or read. B_s is
        // unpadded (BK*BN), matching the book's analysis that only A_s's
        // strided strip-load exhibits bank conflicts.
        __shared__ float A_s[BM * BK_PADDED];
        __shared__ float B_s[BK * BN];
        loadTile(&A[bRow * K + tile * BK], K, M - bRow, K - tile * BK, &A_s[0], BK_PADDED, BM, BK);
        loadTile(&B[tile * BK * N + bCol], N, K - tile * BK, N - bCol, &B_s[0], BN, BK, BN);
        __syncthreads();

        #pragma unroll
        for (unsigned int q = 0; q < NUM_QUADS; ++q) {
            mm(QM, QN, BK, &A_s[rowBaseLocal[q] * BK_PADDED], BK_PADDED, &B_s[colBaseLocal[q]], BN,
               C_r[q]);
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

    mm_tiled_bank_conflict_free_kernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    mm_tiled_bank_conflict_free_kernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, M, N, K);
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

    printf("M=%-5u N=%-5u K=%-5u (BM=%d BN=%d BK=%d, A_s lda=%d): %8.3f ms  [%s]\n", M, N, K, BM,
           BN, BK, BK_PADDED, ms, ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    bool ok = true;
    ok = runTestCase(200, 180, 90) && ok;
    ok = runTestCase(1024, 1024, 1024) && ok;
    ok = runTestCase(131, 173, 67) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
