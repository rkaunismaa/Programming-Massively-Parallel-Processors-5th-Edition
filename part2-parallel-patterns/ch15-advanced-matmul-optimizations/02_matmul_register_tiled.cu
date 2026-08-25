// Chapter 15: Advanced optimizations for matrix multiplication
// §15.4  Register tiling of the input tiles -- mm(), Fig. 15.9.
//
// Builds on 01_matmul_coarsened_larger_tiles.cu (§15.3): same block-level
// (bM x bN) / thread-level (tM x tN) tiling, same clear()/loadTile()/
// writeTile() and main kernel structure (Figs. 15.3-15.5, 15.7). The only
// change is mm(): in 01, each element of a thread's tM x bK / bK x tN
// thread-level input tile is re-read from shared memory once per output
// element that uses it (row/col outer, i inner). §15.4 observes that a
// thread reuses the same input value across many output elements, and that
// each such reuse still pays a shared-memory access -- shared memory is
// much faster than global memory, but still slower than a register.
//
// Fig. 15.9's mm() interchanges the loop nest so the inner (k) dimension
// becomes outermost: for each of the bK "strips" of the thread-level input
// tiles, the thread loads one row-strip of A (a_r[tM]) and one column-strip
// of B (b_r[tN]) from shared memory into registers ONCE, then reuses those
// registers to update every one of the tMxtN output elements before moving
// to the next strip. This removes the redundant shared-memory reads that
// 01's row/col/i loop order pays for.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

#define BM 128
#define BN 128
#define BK 8

#define TM 8
#define TN 8

#define NUM_THREADS_PER_BLOCK 256

// Fig. 15.4: clear() -- unchanged from 01.
__device__ __forceinline__ void clear(float C_r[][TN], unsigned int m, unsigned int n) {
    #pragma unroll
    for (unsigned int row = 0; row < m; ++row) {
        #pragma unroll
        for (unsigned int col = 0; col < n; ++col) {
            C_r[row][col] = 0.0f;
        }
    }
}

// Fig. 15.5: loadTile() -- unchanged from 01.
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
// Fig. 15.9: mm() with register tiling of the input tiles. The outer loop
// (i, line 05 in the book) walks the bK strips of the thread-level input
// tiles. For each strip, a_r (a column of A's thread-level tile) and b_r (a
// row of B's thread-level tile) are loaded from shared memory into
// registers once (loops fully unrolled so a_r/b_r get constant indices and
// are promoted to registers), then reused to update all tM*tN output
// elements in the final loop nest -- each shared-memory element is read
// exactly once per strip, regardless of tM/tN.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mm(unsigned int m, unsigned int n, unsigned int k, const float *a,
                                    unsigned int lda, const float *b, unsigned int ldb,
                                    float c[][TN]) {
    #pragma unroll
    for (unsigned int i = 0; i < k; ++i) {
        // Load A strip to registers.
        float a_r[TM];
        #pragma unroll
        for (unsigned int row = 0; row < m; ++row) {
            a_r[row] = a[row * lda + i];
        }

        // Load B strip to registers.
        float b_r[TN];
        #pragma unroll
        for (unsigned int col = 0; col < n; ++col) {
            b_r[col] = b[i * ldb + col];
        }

        // Compute with strips.
        #pragma unroll
        for (unsigned int row = 0; row < m; ++row) {
            #pragma unroll
            for (unsigned int col = 0; col < n; ++col) {
                c[row][col] += a_r[row] * b_r[col];
            }
        }
    }
}

// Fig. 15.7: writeTile() -- unchanged from 01.
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

// Fig. 15.3: mm_tiled_kernel() -- unchanged from 01 (only mm()'s
// implementation differs).
__global__ void mm_tiled_kernel(const float *A, const float *B, float *C, unsigned int M,
                                 unsigned int N, unsigned int K) {
    unsigned int bRow = blockIdx.y * BM;
    unsigned int bCol = blockIdx.x * BN;

    unsigned int tilesPerBlockX = BN / TN;
    unsigned int ty = threadIdx.x / tilesPerBlockX;
    unsigned int tx = threadIdx.x % tilesPerBlockX;
    unsigned int tRow = ty * TM;
    unsigned int tCol = tx * TN;

    float C_r[TM][TN];
    clear(C_r, TM, TN);

    for (unsigned int tile = 0; tile < (K + BK - 1) / BK; ++tile) {
        __shared__ float A_s[BM * BK];
        __shared__ float B_s[BK * BN];
        loadTile(&A[bRow * K + tile * BK], K, M - bRow, K - tile * BK, &A_s[0], BK, BM, BK);
        loadTile(&B[tile * BK * N + bCol], N, K - tile * BK, N - bCol, &B_s[0], BN, BK, BN);
        __syncthreads();

        mm(TM, TN, BK, &A_s[tRow * BK], BK, &B_s[tCol], BN, C_r);
        __syncthreads();
    }

    float *c = &C[(bRow + tRow) * N + bCol + tCol];
    unsigned int maxRow = (bRow + tRow < M) ? (M - (bRow + tRow)) : 0;
    unsigned int maxCol = (bCol + tCol < N) ? (N - (bCol + tCol)) : 0;
    writeTile(c, N, maxRow, maxCol, C_r, TM, TN);
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
    ok = runTestCase(200, 180, 90) && ok;
    ok = runTestCase(1024, 1024, 1024) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
