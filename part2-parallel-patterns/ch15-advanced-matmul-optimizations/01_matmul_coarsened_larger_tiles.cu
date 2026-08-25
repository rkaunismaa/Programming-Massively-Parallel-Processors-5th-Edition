// Chapter 15: Advanced optimizations for matrix multiplication
// §15.3  Using larger tiles with thread coarsening -- mm_tiled_kernel, Fig.
// 15.3 (main kernel), Fig. 15.4 (clear), Fig. 15.5 (loadTile), Fig. 15.6
// (mm), Fig. 15.7 (writeTile).
//
// §15.2's data reuse analysis shows arithmetic intensity is
// 0.5*m*n/(m+n) operations/byte for an m x n output tile with inner
// dimension k -- independent of k, and growing with larger m, n. Chapter 5's
// tiled matmul used one thread per output element (a TILE_WIDTHxTILE_WIDTH
// == number-of-threads output tile per block). This file instead assigns
// each thread BLOCK a much larger bMxbN "block-level output tile" (here
// 128x128) than it has threads (256, i.e. 16x16), so each individual thread
// must be responsible for a tMxtN "thread-level output tile" (here 8x8) --
// thread coarsening. Enlarging the output tile means the same input panel
// of A (bM x K) and B (K x bN) is reused across many more output elements
// before being reloaded, raising arithmetic intensity and pushing the
// kernel from memory-bound toward compute-bound (§15.2's H100 example: 8
// FLOP/B at 32x32 tiles vs. 32 FLOP/B at 128x128 tiles).
//
// This file reimplements Chapter 5's tiled matmul idea from scratch (no
// cross-chapter include, per this repo's convention) using the book's
// generalized bM/bN/bK block-tile, tM/tN thread-tile parameterization.
// Every accumulation loop over the fixed-size local array C_r is annotated
// #pragma unroll (per §15.3's explanation) so the compiler can prove
// constant indices and promote C_r into registers rather than spilling it
// to local (i.e. global) memory.
//
// The four device functions below (clear, loadTile, mm, writeTile) are
// exact transcriptions of Figs. 15.4-15.7, inlined into the Fig. 15.3 main
// kernel mm_tiled_kernel. mm() here uses the *basic* (non-register-tiled)
// loop order row/col/i -- §15.4's register-tiled variant is 02's subject.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

// Block-level output tile (bM x bN) and inner-dimension tile depth (bK),
// per §15.3's running example and §15.7's occupancy discussion (k can be
// small without hurting arithmetic intensity, since the ratio is
// independent of k; bM=bN=128, bK=8 keeps A_s/B_s at 4 KB each).
#define BM 128
#define BN 128
#define BK 8

// Thread-level output tile (tM x tN): bM x bN divided across a 256-thread
// (16x16-equivalent, addressed as 1D threadIdx.x) block.
#define TM 8
#define TN 8

#define NUM_THREADS_PER_BLOCK 256

// ---------------------------------------------------------------------------
// Fig. 15.4: clear(). Zero-initializes the thread-level output tile. Both
// loops are #pragma unroll'd so C_r is accessed with compile-time-constant
// indices and can be promoted to registers.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void clear(float C_r[][TN], unsigned int m, unsigned int n) {
    #pragma unroll
    for (unsigned int row = 0; row < m; ++row) {
        #pragma unroll
        for (unsigned int col = 0; col < n; ++col) {
            C_r[row][col] = 0.0f;
        }
    }
}

// ---------------------------------------------------------------------------
// Fig. 15.5: loadTile(). Loads an input tile from global memory (T, leading
// dimension lda, valid region maxRow x maxCol) into shared memory (T_s,
// leading dimension ldas, size height x width), zero-padding out-of-bounds
// elements. When the block has fewer threads than tile elements, the tile
// is divided into numSubTiles sub-tiles and loaded one sub-tile at a time.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Fig. 15.6: mm(). Computes the contribution of a pair of thread-level input
// tiles (a: tM x k sub-tile of A_s, b: k x tN sub-tile of B_s) to the
// thread-level output tile c, accumulating in place. Basic loop order:
// output row, output col, then the inner (k) reduction -- each input
// element is re-read from shared memory once per output element that uses
// it (§15.4 removes this redundancy with register tiling).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mm(unsigned int m, unsigned int n, unsigned int k, const float *a,
                                    unsigned int lda, const float *b, unsigned int ldb,
                                    float c[][TN]) {
    #pragma unroll
    for (unsigned int row = 0; row < m; ++row) {
        #pragma unroll
        for (unsigned int col = 0; col < n; ++col) {
            #pragma unroll
            for (unsigned int i = 0; i < k; ++i) {
                c[row][col] += a[row * lda + i] * b[i * ldb + col];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fig. 15.7: writeTile(). Writes the thread-level output tile from
// registers (C_r, size m x n) to global memory (c, leading dimension ldc,
// valid region maxRow x maxCol), skipping out-of-bounds elements.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void writeTile(float *c, unsigned int ldc, unsigned int maxRow,
                                           unsigned int maxCol, float C_r[][TN], unsigned int m,
                                           unsigned int n) {
    #pragma unroll
    for (unsigned int row = 0; row < m; ++row) {
        #pragma unroll
        for (unsigned int col = 0; col < n; ++col) {
            if (row < maxRow && col < maxCol) {
                c[row * ldc + col] = C_r[row][col];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fig. 15.3: mm_tiled_kernel(). Main kernel. Each block computes a bMxbN
// block-level output tile of C; each thread within the block computes a
// tMxtN thread-level output tile, held in registers (C_r), by iterating
// over K in steps of bK, staging each pair of input tiles through shared
// memory (A_s, B_s).
// ---------------------------------------------------------------------------
__global__ void mm_tiled_kernel(const float *A, const float *B, float *C, unsigned int M,
                                 unsigned int N, unsigned int K) {
    // Identify the block's tile.
    unsigned int bRow = blockIdx.y * BM;
    unsigned int bCol = blockIdx.x * BN;

    // Identify the thread's tile within the block.
    unsigned int tilesPerBlockX = BN / TN;
    unsigned int ty = threadIdx.x / tilesPerBlockX;
    unsigned int tx = threadIdx.x % tilesPerBlockX;
    unsigned int tRow = ty * TM;
    unsigned int tCol = tx * TN;

    // Initialize the output tile.
    float C_r[TM][TN];
    clear(C_r, TM, TN);

    // Iterate over input tiles.
    for (unsigned int tile = 0; tile < (K + BK - 1) / BK; ++tile) {
        // Load A and B tiles to shared memory.
        __shared__ float A_s[BM * BK];
        __shared__ float B_s[BK * BN];
        loadTile(&A[bRow * K + tile * BK], K, M - bRow, K - tile * BK, &A_s[0], BK, BM, BK);
        loadTile(&B[tile * BK * N + bCol], N, K - tile * BK, N - bCol, &B_s[0], BN, BK, BN);
        __syncthreads();

        // Compute with shared memory tiles.
        mm(TM, TN, BK, &A_s[tRow * BK], BK, &B_s[tCol], BN, C_r);
        __syncthreads();
    }

    // Write output tile.
    float *c = &C[(bRow + tRow) * N + bCol + tCol];
    unsigned int maxRow = (bRow + tRow < M) ? (M - (bRow + tRow)) : 0;
    unsigned int maxCol = (bCol + tCol < N) ? (N - (bCol + tCol)) : 0;
    writeTile(c, N, maxRow, maxCol, C_r, TM, TN);
}

// CPU reference: standard row-major A (MxK) times B (KxN) -> C (MxN).
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

// Runs mm_tiled_kernel once (with a discarded warm-up launch first) and
// returns the timed kernel duration in ms.
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

    // Warm-up launch (discarded).
    mm_tiled_kernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    mm_tiled_kernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    float ms = timer.stopAndGetMs();

    CUDA_CHECK(cudaMemcpy(C_h, C_d, sizeC, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));

    return ms;
}

// Runs one (M, N, K) test case: builds inputs, computes the CPU reference,
// launches the kernel, and checks agreement.
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

    printf("M=%-5u N=%-5u K=%-5u (BM=%d BN=%d BK=%d, TM=%d TN=%d): %8.3f ms  [%s]\n", M, N, K, BM,
           BN, BK, TM, TN, ms, ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    bool ok = true;
    // Non-multiple-of-tile-dims case: exercises loadTile/writeTile boundary
    // checks along both M and N, and a K not a multiple of BK.
    ok = runTestCase(200, 180, 90) && ok;
    // Exact-multiple performance case (BM=128, BN=128, BK=8 all divide
    // evenly), large enough for a meaningful timing.
    ok = runTestCase(1024, 1024, 1024) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
