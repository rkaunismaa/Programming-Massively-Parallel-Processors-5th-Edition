// Chapter 3: Multidimensional grids and data
// §3.4  Matrix multiplication -- matrixMulKernel, Fig. 3.11
//
// M, N, P are square Width x Width matrices linearized in row-major order.
// Each thread computes one output element as an inner (dot) product:
//   P_row,col = sum_k  M_row,k * N_k,col ,   k = 0 .. Width-1
// using the same thread-to-output-element mapping as colorToGrayscaleConversion
// and blurKernel:
//   row = blockIdx.y*blockDim.y + threadIdx.y
//   col = blockIdx.x*blockDim.x + threadIdx.x
// This is the book's "naive" one-thread-per-output-element kernel, with no
// use of shared memory (that optimization comes in Chapter 5).

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

// ---------------------------------------------------------------------------
// §3.4, Fig. 3.11: matrixMulKernel.
// M is row-major, so the kth element of row `row` is M[row*Width+k]. N is
// also row-major, so walking down column `col` means skipping an entire row
// per step: the kth element of column `col` is N[k*Width+col]. Pvalue
// accumulates the inner product before being written to P[row*Width+col].
// ---------------------------------------------------------------------------
__global__ void matrixMulKernel(const float *M, const float *N, float *P, int Width) {
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

// CPU reference: identical inner-product formula and loop order.
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

float runMatmul(const float *M_h, const float *N_h, float *P_h, int Width) {
    float *M_d, *N_d, *P_d;
    size_t size = static_cast<size_t>(Width) * Width * sizeof(float);

    CUDA_CHECK(cudaMalloc((void **)&M_d, size));
    CUDA_CHECK(cudaMalloc((void **)&N_d, size));
    CUDA_CHECK(cudaMalloc((void **)&P_d, size));

    CUDA_CHECK(cudaMemcpy(M_d, M_h, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(N_d, N_h, size, cudaMemcpyHostToDevice));

    // §3.4 maps threads to output elements exactly as §3.2 mapped them to
    // pixels: a fixed 16x16 dimBlock and a dimGrid sized by ceiling division
    // of Width by 16 in both dimensions.
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid((Width + 15) / 16, (Width + 15) / 16, 1);

    GpuTimer timer;
    timer.start();
    matrixMulKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, Width);
    CUDA_CHECK(cudaGetLastError());
    float ms = timer.stopAndGetMs();

    CUDA_CHECK(cudaMemcpy(P_h, P_d, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(M_d));
    CUDA_CHECK(cudaFree(N_d));
    CUDA_CHECK(cudaFree(P_d));

    return ms;
}

int main() {
    const int Width = 500;  // not a multiple of the 16x16 block size

    std::vector<float> M_h(static_cast<size_t>(Width) * Width);
    std::vector<float> N_h(static_cast<size_t>(Width) * Width);
    std::vector<float> P_ref(static_cast<size_t>(Width) * Width);
    std::vector<float> P_h(static_cast<size_t>(Width) * Width);

    for (int i = 0; i < Width * Width; ++i) {
        M_h[i] = static_cast<float>(i % 13) * 0.1f - 0.6f;
        N_h[i] = static_cast<float>(i % 7) * 0.2f - 0.6f;
    }

    // CPU reference (§3.4 inner-product formula).
    matrixMul_h(M_h.data(), N_h.data(), P_ref.data(), Width);

    // GPU version (§3.4, Fig. 3.11).
    float ms = runMatmul(M_h.data(), N_h.data(), P_h.data(), Width);

    bool ok = true;
    for (int i = 0; i < Width * Width; ++i) {
        if (!nearlyEqual(P_h[i], P_ref[i])) {
            ok = false;
            fprintf(stderr, "Mismatch at i=%d: gpu=%f cpu=%f\n", i, P_h[i], P_ref[i]);
            break;
        }
    }

    dim3 dimGrid((Width + 15) / 16, (Width + 15) / 16, 1);
    printf("Width = %d, dimBlock=(16,16,1), dimGrid=(%d,%d,1)\n",
           Width, dimGrid.x, dimGrid.y);
    printf("GPU matrixMulKernel time: %.3f ms\n", ms);
    printf("%s\n", ok ? "PASS" : "FAIL");

    return ok ? 0 : 1;
}
