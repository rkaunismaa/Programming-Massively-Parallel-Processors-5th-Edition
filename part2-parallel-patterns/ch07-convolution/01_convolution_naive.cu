// Chapter 7: Convolution and constant memory
// §7.1  Background -- 1D convolution is used to introduce the definitions
//       (filter/radius, ghost cells), but the chapter's actual code examples
//       (§7.2 onward) are all 2D convolution over image-shaped arrays, "and
//       the reader is encouraged to adapt these code examples to 1D and 3D
//       as exercises." This file, like the rest of the chapter, implements
//       2D convolution.
// §7.2  Parallel convolution -- a basic kernel -- convolution_2D_basic_kernel
//       (Fig. 7.7)
//
// The mask (convolution filter) F lives in ordinary global memory and is
// passed to the kernel as a pointer, exactly like N and P. Each thread is
// assigned one output element P[outRow][outCol], matching its (x, y) thread
// index within the grid (Fig. 7.6) -- the same parallelization scheme as the
// ColorToGrayScaleConversion kernel of Chapter 3.
//
// For each output element, the thread walks the (2r+1) x (2r+1) filter and
// accumulates a weighted sum of the corresponding input neighborhood into a
// register (Pvalue), only writing to global memory once at the end. Input
// elements that fall outside the N array (the "ghost cells" of §7.1) are
// treated as 0 by simply skipping their multiply-accumulate -- the bounds
// check on lines 09-10 of Fig. 7.7. Because different threads near the four
// edges of P skip a different number of times, this introduces some control
// divergence, but the book notes this effect is "modest to insignificant"
// for the large images and small filters convolution is normally applied to
// (§7.2).

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

#define BLOCK_DIM 16

// ---------------------------------------------------------------------------
// §7.2, Fig. 7.7: convolution_2D_basic_kernel. F is a (2r+1) x (2r+1) filter
// stored row-major in global memory. Ghost cells (input neighbors outside
// [0, width) x [0, height)) contribute 0 and are simply skipped.
// ---------------------------------------------------------------------------
__global__ void convolution_2D_basic_kernel(const float *N, const float *F, float *P, int r, int width, int height) {
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;

    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < 2 * r + 1; ++fRow) {
        for (int fCol = 0; fCol < 2 * r + 1; ++fCol) {
            int inRow = outRow - r + fRow;
            int inCol = outCol - r + fCol;
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                Pvalue += F[fRow * (2 * r + 1) + fCol] * N[inRow * width + inCol];
            }
        }
    }

    if (outRow < height && outCol < width) {
        P[outRow * width + outCol] = Pvalue;
    }
}

// CPU reference: identical weighted-sum-with-zero-ghost-cells formula as the
// kernel, used to validate the GPU result.
void convolution2D_h(const float *N, const float *F, float *P, int r, int width, int height) {
    for (int outRow = 0; outRow < height; ++outRow) {
        for (int outCol = 0; outCol < width; ++outCol) {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < 2 * r + 1; ++fRow) {
                for (int fCol = 0; fCol < 2 * r + 1; ++fCol) {
                    int inRow = outRow - r + fRow;
                    int inCol = outCol - r + fCol;
                    if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                        Pvalue += F[fRow * (2 * r + 1) + fCol] * N[inRow * width + inCol];
                    }
                }
            }
            P[outRow * width + outCol] = Pvalue;
        }
    }
}

// Runs the basic kernel once (with a discarded warm-up launch first, per the
// project's timing-methodology precedent) and returns the timed kernel
// duration in ms. P_h receives the copied-back result.
float runBasicConvolution(const float *N_h, const float *F_h, float *P_h, int r, int width, int height) {
    size_t nSize = static_cast<size_t>(width) * height * sizeof(float);
    size_t fSize = static_cast<size_t>(2 * r + 1) * (2 * r + 1) * sizeof(float);

    float *N_d, *F_d, *P_d;
    CUDA_CHECK(cudaMalloc((void **)&N_d, nSize));
    CUDA_CHECK(cudaMalloc((void **)&F_d, fSize));
    CUDA_CHECK(cudaMalloc((void **)&P_d, nSize));
    CUDA_CHECK(cudaMemcpy(N_d, N_h, nSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(F_d, F_h, fSize, cudaMemcpyHostToDevice));

    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM, 1);
    dim3 dimGrid((width + BLOCK_DIM - 1) / BLOCK_DIM, (height + BLOCK_DIM - 1) / BLOCK_DIM, 1);

    // Warm-up launch (discarded) so PTX->SASS JIT cost isn't folded into the
    // timed measurement.
    convolution_2D_basic_kernel<<<dimGrid, dimBlock>>>(N_d, F_d, P_d, r, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    convolution_2D_basic_kernel<<<dimGrid, dimBlock>>>(N_d, F_d, P_d, r, width, height);
    CUDA_CHECK(cudaGetLastError());
    float ms = timer.stopAndGetMs();

    CUDA_CHECK(cudaMemcpy(P_h, P_d, nSize, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(N_d));
    CUDA_CHECK(cudaFree(F_d));
    CUDA_CHECK(cudaFree(P_d));

    return ms;
}

// Runs one (width, height, r) test case: builds N and a filter F, computes
// the CPU reference, launches the kernel, and checks agreement.
bool runTestCase(int width, int height, int r) {
    size_t nCount = static_cast<size_t>(width) * height;
    size_t fCount = static_cast<size_t>(2 * r + 1) * (2 * r + 1);

    std::vector<float> N_h(nCount), P_ref(nCount), P_h(nCount);
    std::vector<float> F_h(fCount);

    for (size_t i = 0; i < nCount; ++i) {
        N_h[i] = static_cast<float>(i % 13) * 0.1f - 0.6f;
    }
    for (size_t i = 0; i < fCount; ++i) {
        F_h[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.1f;
    }

    convolution2D_h(N_h.data(), F_h.data(), P_ref.data(), r, width, height);
    float ms = runBasicConvolution(N_h.data(), F_h.data(), P_h.data(), r, width, height);

    bool ok = true;
    for (size_t i = 0; i < nCount; ++i) {
        if (!nearlyEqual(P_h[i], P_ref[i])) {
            ok = false;
            fprintf(stderr, "Mismatch at width=%d height=%d r=%d i=%zu: gpu=%f cpu=%f\n", width, height, r, i,
                    P_h[i], P_ref[i]);
            break;
        }
    }

    printf("width=%-4d height=%-4d r=%d: %.3f ms  [%s]\n", width, height, r, ms, ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    bool ok = true;
    // Deliberately mix sizes that are and aren't multiples of BLOCK_DIM (16)
    // so the ghost-cell handling on all four edges is exercised, plus a
    // couple of filter radii, and one larger case for representative timing.
    ok = runTestCase(67, 51, 2) && ok;
    ok = runTestCase(33, 33, 1) && ok;
    ok = runTestCase(300, 200, 3) && ok;
    ok = runTestCase(1024, 1024, 2) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
