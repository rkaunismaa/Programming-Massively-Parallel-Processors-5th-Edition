// Chapter 6: Performance considerations
// §6.5  Thread coarsening, applied to the tiled matrix multiplication kernel
// (the example §6.5 names explicitly: "Another example where thread
// coarsening could be beneficial is the tiled matrix multiplication kernel
// in Chapter 5 ... Using fewer thread blocks and having each block process
// larger output tiles enables each input tile to be loaded fewer times.")
//
// This file reimplements Chapter 5's tiled matmul locally (no cross-chapter
// include, per the brief) and applies a coarsening factor COARSE_FACTOR: each
// thread block still loads one M tile per phase, but now reuses that single
// M tile across COARSE_FACTOR separate N tiles / output tiles laid out
// side by side along the column direction, instead of one thread block
// loading its own private copy of the M tile for each output tile. This is
// exactly the redundant-load reduction §6.5 describes: COARSE_FACTOR fewer
// thread blocks means the M tile is fetched from global memory
// COARSE_FACTOR times less often for the same output.
//
// Coarsening pitfall #2 from §6.5 (not exposing enough parallelism / partial
// waves) is why COARSE_FACTOR is kept modest (4) and Width large enough
// (1024) that both the coarsened and uncoarsened grids comfortably exceed
// one wave on this hardware.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

#define TILE_WIDTH 32
#define COARSE_FACTOR 4

// ---------------------------------------------------------------------------
// Uncoarsened baseline: Chapter 5's tiled matmul (Fig. 5.9), reimplemented
// locally. One thread block computes one TILE_WIDTH x TILE_WIDTH output tile.
// ---------------------------------------------------------------------------
__global__ void matmulTiledKernel(const float *M, const float *N, float *P, int Width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0f;
    for (int ph = 0; ph < Width / TILE_WIDTH; ++ph) {
        Mds[ty][tx] = M[Row * Width + ph * TILE_WIDTH + tx];
        Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }

    P[Row * Width + Col] = Pvalue;
}

// ---------------------------------------------------------------------------
// §6.5: coarsened version. One thread block computes COARSE_FACTOR
// TILE_WIDTH x TILE_WIDTH output tiles laid out side by side along the
// column direction. Each phase loads the M tile ONCE and reuses it across
// all COARSE_FACTOR N-tile loads, each accumulated into its own register
// (Pvalue[c]) -- this is the register tiling from §5.2/§6.8 combined with
// the coarser thread-block granularity from §6.5.
// ---------------------------------------------------------------------------
__global__ void matmulCoarsenedKernel(const float *M, const float *N, float *P, int Width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int ColStart = bx * TILE_WIDTH * COARSE_FACTOR + tx;

    float Pvalue[COARSE_FACTOR];
    for (int c = 0; c < COARSE_FACTOR; ++c) Pvalue[c] = 0.0f;

    for (int ph = 0; ph < Width / TILE_WIDTH; ++ph) {
        // Load the M tile once per phase -- shared across all COARSE_FACTOR
        // output tiles this block is responsible for.
        Mds[ty][tx] = M[Row * Width + ph * TILE_WIDTH + tx];

        for (int c = 0; c < COARSE_FACTOR; ++c) {
            int Col = ColStart + c * TILE_WIDTH;

            Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];
            __syncthreads();

            for (int k = 0; k < TILE_WIDTH; ++k) {
                Pvalue[c] += Mds[ty][k] * Nds[k][tx];
            }
            __syncthreads();
        }
    }

    for (int c = 0; c < COARSE_FACTOR; ++c) {
        int Col = ColStart + c * TILE_WIDTH;
        P[Row * Width + Col] = Pvalue[c];
    }
}

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
    const int Width = 1024;  // multiple of TILE_WIDTH*COARSE_FACTOR (32*4=128)
    if (Width % (TILE_WIDTH * COARSE_FACTOR) != 0) {
        fprintf(stderr, "Width must be a multiple of TILE_WIDTH*COARSE_FACTOR for this file\n");
        return 1;
    }

    size_t count = static_cast<size_t>(Width) * Width;
    size_t size = count * sizeof(float);

    std::vector<float> M_h(count), N_h(count), P_ref(count);
    std::vector<float> P_tiled_h(count), P_coarsened_h(count);

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
    // Uncoarsened grid: one block per output tile.
    dim3 dimGridUncoarsened(Width / TILE_WIDTH, Width / TILE_WIDTH, 1);
    // Coarsened grid: COARSE_FACTOR fewer blocks along x -- each block now
    // covers COARSE_FACTOR output tiles. Same block dim, same total output,
    // same M/N/P data -- the reduced block count *is* the optimization being
    // demonstrated, not an unfair setup.
    dim3 dimGridCoarsened(Width / (TILE_WIDTH * COARSE_FACTOR), Width / TILE_WIDTH, 1);

    // Warm up both kernels once each (discarded) before timing.
    matmulTiledKernel<<<dimGridUncoarsened, dimBlock>>>(M_d, N_d, P_d, Width);
    CUDA_CHECK(cudaGetLastError());
    matmulCoarsenedKernel<<<dimGridCoarsened, dimBlock>>>(M_d, N_d, P_d, Width);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    matmulTiledKernel<<<dimGridUncoarsened, dimBlock>>>(M_d, N_d, P_d, Width);
    CUDA_CHECK(cudaGetLastError());
    float tiled_ms = timer.stopAndGetMs();
    CUDA_CHECK(cudaMemcpy(P_tiled_h.data(), P_d, size, cudaMemcpyDeviceToHost));

    timer.start();
    matmulCoarsenedKernel<<<dimGridCoarsened, dimBlock>>>(M_d, N_d, P_d, Width);
    CUDA_CHECK(cudaGetLastError());
    float coarsened_ms = timer.stopAndGetMs();
    CUDA_CHECK(cudaMemcpy(P_coarsened_h.data(), P_d, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(M_d));
    CUDA_CHECK(cudaFree(N_d));
    CUDA_CHECK(cudaFree(P_d));

    bool tiled_ok = true, coarsened_ok = true;
    for (size_t i = 0; i < count; ++i) {
        if (tiled_ok && !nearlyEqual(P_tiled_h[i], P_ref[i])) {
            tiled_ok = false;
            fprintf(stderr, "Uncoarsened mismatch at i=%zu: gpu=%f cpu=%f\n", i, P_tiled_h[i], P_ref[i]);
        }
        if (coarsened_ok && !nearlyEqual(P_coarsened_h[i], P_ref[i])) {
            coarsened_ok = false;
            fprintf(stderr, "Coarsened mismatch at i=%zu: gpu=%f cpu=%f\n", i, P_coarsened_h[i], P_ref[i]);
        }
        if (!tiled_ok && !coarsened_ok) break;
    }

    printf("Width = %d, TILE_WIDTH = %d, COARSE_FACTOR = %d\n", Width, TILE_WIDTH, COARSE_FACTOR);
    printf("Uncoarsened (Ch.5 tiled, dimGrid=(%d,%d,1)) kernel time: %.3f ms  [%s]\n",
           dimGridUncoarsened.x, dimGridUncoarsened.y, tiled_ms, tiled_ok ? "match" : "MISMATCH");
    printf("Coarsened   (§6.5, dimGrid=(%d,%d,1))       kernel time: %.3f ms  [%s]\n",
           dimGridCoarsened.x, dimGridCoarsened.y, coarsened_ms, coarsened_ok ? "match" : "MISMATCH");
    printf("Speedup (uncoarsened/coarsened): %.2fx\n", tiled_ms / coarsened_ms);

    bool ok = tiled_ok && coarsened_ok;
    printf("%s\n", ok ? "PASS" : "FAIL");

    return ok ? 0 : 1;
}
