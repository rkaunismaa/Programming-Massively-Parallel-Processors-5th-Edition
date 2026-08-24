// Chapter 5: Memory architecture and data locality
// §5.4  A tiled matrix multiplication kernel -- matrixMulTiledKernel, Fig. 5.9
//
// Square Width x Width matrices M, N, P, linearized in row-major order, with
// Width an exact multiple of TILE_WIDTH (the simplifying assumption §5.4
// makes explicit at the end of the section; §5.5 removes it -- see
// 02_tiled_matmul_boundary_checked.cu).
//
// The book's key idea (§5.3): the naive kernel of Fig. 3.11 has every thread
// re-read a full row of M and a full column of N from global memory, but
// within a block those rows/columns overlap heavily across threads. Tiling
// has all threads of a block collaboratively stage a TILE_WIDTH x TILE_WIDTH
// tile of M and one of N into __shared__ memory once, then every thread in
// the block reuses those on-chip tiles TILE_WIDTH times each before moving to
// the next tile pair ("phase"). This is strip-mining the Width-long dot
// product into Width/TILE_WIDTH phases (§5.4), and it cuts global memory
// traffic for M and N by a factor of TILE_WIDTH.
//
// This file also runs the §3.4 naive kernel (no shared memory) on the same
// inputs for timing context, per the brief -- the comparison is informative
// only: PASS/FAIL is decided solely by agreement with the CPU reference.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

// §5.6 gives this exact value in its worked example ("#define TILE_WIDTH 32"),
// so we use it here too.
#define TILE_WIDTH 32

// ---------------------------------------------------------------------------
// §5.4, Fig. 5.9: matrixMulTiledKernel. Assumes Width is a multiple of
// TILE_WIDTH so every tile load and every P write is in-bounds unconditionally.
// ---------------------------------------------------------------------------
__global__ void matrixMulTiledKernel(const float *M, const float *N, float *P, int Width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // Row/Col of the P element this thread is responsible for (§5.4, Fig. 5.10).
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0f;

    // Strip-mine the Width-long dot product into Width/TILE_WIDTH phases.
    for (int ph = 0; ph < Width / TILE_WIDTH; ++ph) {
        // Collaboratively load one tile of M and one tile of N: each thread
        // loads exactly one element of each, tx/ty positions from the tile's
        // starting column/row (§5.4, lines 19-20 of Fig. 5.9).
        Mds[ty][tx] = M[Row * Width + ph * TILE_WIDTH + tx];
        Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];

        // Read-after-write barrier: wait until every thread's loads have
        // landed in shared memory before any thread starts consuming them.
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }

        // Write-after-read barrier: wait until every thread is done reading
        // this phase's tiles before the next iteration overwrites them.
        __syncthreads();
    }

    P[Row * Width + Col] = Pvalue;
}

// ---------------------------------------------------------------------------
// §3.4, Fig. 3.11: the naive one-thread-per-output-element kernel, with no
// shared memory, run here only for timing context (§5.1/§5.4 motivate tiling
// by contrasting against exactly this kernel's memory traffic).
// ---------------------------------------------------------------------------
__global__ void matrixMulNaiveKernel(const float *M, const float *N, float *P, int Width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < Width && col < Width) {
        float Pvalue = 0.0f;
        for (int k = 0; k < Width; ++k) {
            Pvalue += M[row * Width + k] * N[k * Width + col];
        }
        P[row * Width + col] = Pvalue;
    }
}

// CPU reference: identical inner-product formula and loop order as §3.4.
void matrixMul_h(const float *M, const float *N, float *P, int Width) {
    for (int row = 0; row < Width; ++row) {
        for (int col = 0; col < Width; ++col) {
            float Pvalue = 0.0f;
            for (int k = 0; k < Width; ++k) {
                Pvalue += M[row * Width + k] * N[k * Width + col];
            }
            P[row * Width + col] = Pvalue;
        }
    }
}

int main() {
    const int Width = 1024;  // exact multiple of TILE_WIDTH (32)

    if (Width % TILE_WIDTH != 0) {
        fprintf(stderr, "Width must be a multiple of TILE_WIDTH for this file\n");
        return 1;
    }

    size_t count = static_cast<size_t>(Width) * Width;
    size_t size = count * sizeof(float);

    std::vector<float> M_h(count), N_h(count), P_ref(count);
    std::vector<float> P_tiled_h(count), P_naive_h(count);

    for (size_t i = 0; i < count; ++i) {
        M_h[i] = static_cast<float>(i % 13) * 0.1f - 0.6f;
        N_h[i] = static_cast<float>(i % 7) * 0.2f - 0.6f;
    }

    printf("Computing CPU reference (Width=%d, %zu elements)...\n", Width, count);
    matrixMul_h(M_h.data(), N_h.data(), P_ref.data(), Width);

    float *M_d, *N_d, *P_d;
    CUDA_CHECK(cudaMalloc((void **)&M_d, size));
    CUDA_CHECK(cudaMalloc((void **)&N_d, size));
    CUDA_CHECK(cudaMalloc((void **)&P_d, size));
    CUDA_CHECK(cudaMemcpy(M_d, M_h.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(N_d, N_h.data(), size, cudaMemcpyHostToDevice));

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid(Width / TILE_WIDTH, Width / TILE_WIDTH, 1);

    // Warm up both kernels once each (discarded) before timing, so the
    // one-time PTX->SASS JIT cost doesn't get attributed to whichever kernel
    // happens to launch first (Task 3's lesson, per the chapter brief).
    matrixMulTiledKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, Width);
    CUDA_CHECK(cudaGetLastError());
    matrixMulNaiveKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, Width);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run: tiled kernel (§5.4, Fig. 5.9).
    GpuTimer timer;
    timer.start();
    matrixMulTiledKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, Width);
    CUDA_CHECK(cudaGetLastError());
    float tiled_ms = timer.stopAndGetMs();
    CUDA_CHECK(cudaMemcpy(P_tiled_h.data(), P_d, size, cudaMemcpyDeviceToHost));

    // Timed run: naive kernel (§3.4, Fig. 3.11), same inputs, same launch config.
    timer.start();
    matrixMulNaiveKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, Width);
    CUDA_CHECK(cudaGetLastError());
    float naive_ms = timer.stopAndGetMs();
    CUDA_CHECK(cudaMemcpy(P_naive_h.data(), P_d, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(M_d));
    CUDA_CHECK(cudaFree(N_d));
    CUDA_CHECK(cudaFree(P_d));

    bool tiled_ok = true, naive_ok = true;
    for (size_t i = 0; i < count; ++i) {
        if (tiled_ok && !nearlyEqual(P_tiled_h[i], P_ref[i])) {
            tiled_ok = false;
            fprintf(stderr, "Tiled mismatch at i=%zu: gpu=%f cpu=%f\n", i, P_tiled_h[i], P_ref[i]);
        }
        if (naive_ok && !nearlyEqual(P_naive_h[i], P_ref[i])) {
            naive_ok = false;
            fprintf(stderr, "Naive mismatch at i=%zu: gpu=%f cpu=%f\n", i, P_naive_h[i], P_ref[i]);
        }
        if (!tiled_ok && !naive_ok) break;
    }

    printf("Width = %d, TILE_WIDTH = %d, dimBlock=(%d,%d,1), dimGrid=(%d,%d,1)\n",
           Width, TILE_WIDTH, dimBlock.x, dimBlock.y, dimGrid.x, dimGrid.y);
    printf("Tiled  (§5.4, shared memory) kernel time: %.3f ms  [%s]\n",
           tiled_ms, tiled_ok ? "match" : "MISMATCH");
    printf("Naive  (§3.4, global memory only) kernel time: %.3f ms  [%s]\n",
           naive_ms, naive_ok ? "match" : "MISMATCH");
    printf("Speedup (naive/tiled): %.2fx\n", naive_ms / tiled_ms);

    bool ok = tiled_ok && naive_ok;
    printf("%s\n", ok ? "PASS" : "FAIL");

    return ok ? 0 : 1;
}
