// Chapter 6: Performance considerations
// §6.6  Loop unrolling -- "#pragma unroll N" before a loop instructs the
// compiler to unroll it by a factor of N; "setting the unrolling factor to 1
// instructs the compiler not to perform any unrolling."
//
// This file reimplements Chapter 5's tiled matmul locally (no cross-chapter
// include) as two kernels that are IDENTICAL except for one pragma on the
// inner phase-accumulation loop (the k-loop over TILE_WIDTH products, the
// book's own example of a loop with "a small and constant loop bound" -- see
// the coarsening-loop discussion later in §6.6, which applies the same
// reasoning to exactly this kind of loop):
//   - matmulUnrolledKernel:   #pragma unroll        (full unroll -- TILE_WIDTH
//                                                     is a compile-time constant)
//   - matmulNotUnrolledKernel: #pragma unroll 1      (explicitly disabled, per
//                                                     the book's stated semantics)

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

#define TILE_WIDTH 32

// ---------------------------------------------------------------------------
// §6.6: inner k-loop fully unrolled.
// ---------------------------------------------------------------------------
__global__ void matmulUnrolledKernel(const float *M, const float *N, float *P, int Width) {
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

#pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }

    P[Row * Width + Col] = Pvalue;
}

// ---------------------------------------------------------------------------
// §6.6: inner k-loop unrolling explicitly disabled ("#pragma unroll 1").
// Otherwise byte-for-byte identical to matmulUnrolledKernel above.
// ---------------------------------------------------------------------------
__global__ void matmulNotUnrolledKernel(const float *M, const float *N, float *P, int Width) {
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

#pragma unroll 1
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }

    P[Row * Width + Col] = Pvalue;
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
    const int Width = 1024;  // exact multiple of TILE_WIDTH
    if (Width % TILE_WIDTH != 0) {
        fprintf(stderr, "Width must be a multiple of TILE_WIDTH for this file\n");
        return 1;
    }

    size_t count = static_cast<size_t>(Width) * Width;
    size_t size = count * sizeof(float);

    std::vector<float> M_h(count), N_h(count), P_ref(count);
    std::vector<float> P_unrolled_h(count), P_not_unrolled_h(count);

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

    // Warm up both kernels once each (discarded) before timing, same
    // launch config for both sides.
    matmulUnrolledKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, Width);
    CUDA_CHECK(cudaGetLastError());
    matmulNotUnrolledKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, Width);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    matmulUnrolledKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, Width);
    CUDA_CHECK(cudaGetLastError());
    float unrolled_ms = timer.stopAndGetMs();
    CUDA_CHECK(cudaMemcpy(P_unrolled_h.data(), P_d, size, cudaMemcpyDeviceToHost));

    timer.start();
    matmulNotUnrolledKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, Width);
    CUDA_CHECK(cudaGetLastError());
    float not_unrolled_ms = timer.stopAndGetMs();
    CUDA_CHECK(cudaMemcpy(P_not_unrolled_h.data(), P_d, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(M_d));
    CUDA_CHECK(cudaFree(N_d));
    CUDA_CHECK(cudaFree(P_d));

    bool unrolled_ok = true, not_unrolled_ok = true;
    for (size_t i = 0; i < count; ++i) {
        if (unrolled_ok && !nearlyEqual(P_unrolled_h[i], P_ref[i])) {
            unrolled_ok = false;
            fprintf(stderr, "Unrolled mismatch at i=%zu: gpu=%f cpu=%f\n", i, P_unrolled_h[i], P_ref[i]);
        }
        if (not_unrolled_ok && !nearlyEqual(P_not_unrolled_h[i], P_ref[i])) {
            not_unrolled_ok = false;
            fprintf(stderr, "Not-unrolled mismatch at i=%zu: gpu=%f cpu=%f\n", i, P_not_unrolled_h[i], P_ref[i]);
        }
        if (!unrolled_ok && !not_unrolled_ok) break;
    }

    printf("Width = %d, TILE_WIDTH = %d\n", Width, TILE_WIDTH);
    printf("#pragma unroll   (§6.6, full unroll) kernel time: %.3f ms  [%s]\n",
           unrolled_ms, unrolled_ok ? "match" : "MISMATCH");
    printf("#pragma unroll 1 (§6.6, no unroll)    kernel time: %.3f ms  [%s]\n",
           not_unrolled_ms, not_unrolled_ok ? "match" : "MISMATCH");
    printf("Speedup (no-unroll/unroll): %.2fx\n", not_unrolled_ms / unrolled_ms);

    bool ok = unrolled_ok && not_unrolled_ok;
    printf("%s\n", ok ? "PASS" : "FAIL");

    return ok ? 0 : 1;
}
