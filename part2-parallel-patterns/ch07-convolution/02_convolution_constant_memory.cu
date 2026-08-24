// Chapter 7: Convolution and constant memory
// §7.4  Constant memory and caching -- convolution_2D_const_mem_kernel
//       (Fig. 7.9)
//
// Identical kernel to 01_convolution_naive.cu's Fig. 7.7 (§7.2), with one
// change: the filter F is no longer passed in as a pointer parameter. It is
// declared as a __constant__ global array, sized by the compile-time
// FILTER_RADIUS, and populated once from the host with cudaMemcpyToSymbol
// before the kernel launches. §7.4 gives three reasons F is an ideal fit for
// constant memory: it's small (radius <= 7 for most filters), it never
// changes during kernel execution, and every thread reads it in the same
// order -- so the hardware constant cache can serve nearly all accesses to
// F without any DRAM traffic, roughly doubling arithmetic intensity versus
// the global-memory-filter kernel (0.25 -> 0.5 FLOP/B, §7.4).

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

#define FILTER_RADIUS 2
#define BLOCK_DIM 16

// §7.4: the filter lives in constant memory, declared as a plain global
// variable outside any function, sized (2*FILTER_RADIUS+1)^2.
__constant__ float F[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];

// ---------------------------------------------------------------------------
// §7.4, Fig. 7.9: convolution_2D_const_mem_kernel. Same structure as the
// basic kernel of Fig. 7.7, but F is read as a global constant-memory array
// rather than through a pointer parameter.
// ---------------------------------------------------------------------------
__global__ void convolution_2D_const_mem_kernel(const float *N, float *P, int width, int height) {
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;

    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; ++fRow) {
        for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; ++fCol) {
            int inRow = outRow - FILTER_RADIUS + fRow;
            int inCol = outCol - FILTER_RADIUS + fCol;
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                Pvalue += F[fRow][fCol] * N[inRow * width + inCol];
            }
        }
    }

    if (outRow < height && outCol < width) {
        P[outRow * width + outCol] = Pvalue;
    }
}

// CPU reference: same weighted-sum-with-zero-ghost-cells formula, reading
// the filter from a flat host array (row-major, same layout as F_h below).
void convolution2D_h(const float *N, const float *F_h, float *P, int width, int height) {
    for (int outRow = 0; outRow < height; ++outRow) {
        for (int outCol = 0; outCol < width; ++outCol) {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; ++fRow) {
                for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; ++fCol) {
                    int inRow = outRow - FILTER_RADIUS + fRow;
                    int inCol = outCol - FILTER_RADIUS + fCol;
                    if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                        Pvalue += F_h[fRow * (2 * FILTER_RADIUS + 1) + fCol] * N[inRow * width + inCol];
                    }
                }
            }
            P[outRow * width + outCol] = Pvalue;
        }
    }
}

// Runs the constant-memory kernel once (with a discarded warm-up launch
// first) and returns the timed kernel duration in ms. P_h receives the
// copied-back result. F_h is copied into the __constant__ symbol F via
// cudaMemcpyToSymbol, per §7.4.
float runConstMemConvolution(const float *N_h, const float *F_h, float *P_h, int width, int height) {
    size_t nSize = static_cast<size_t>(width) * height * sizeof(float);
    size_t fSize = static_cast<size_t>(2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1) * sizeof(float);

    float *N_d, *P_d;
    CUDA_CHECK(cudaMalloc((void **)&N_d, nSize));
    CUDA_CHECK(cudaMalloc((void **)&P_d, nSize));
    CUDA_CHECK(cudaMemcpy(N_d, N_h, nSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(F, F_h, fSize));

    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM, 1);
    dim3 dimGrid((width + BLOCK_DIM - 1) / BLOCK_DIM, (height + BLOCK_DIM - 1) / BLOCK_DIM, 1);

    // Warm-up launch (discarded) so PTX->SASS JIT cost isn't folded into the
    // timed measurement.
    convolution_2D_const_mem_kernel<<<dimGrid, dimBlock>>>(N_d, P_d, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    convolution_2D_const_mem_kernel<<<dimGrid, dimBlock>>>(N_d, P_d, width, height);
    CUDA_CHECK(cudaGetLastError());
    float ms = timer.stopAndGetMs();

    CUDA_CHECK(cudaMemcpy(P_h, P_d, nSize, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(N_d));
    CUDA_CHECK(cudaFree(P_d));

    return ms;
}

// Runs one (width, height) test case with a fixed FILTER_RADIUS-2 filter:
// builds N and F, computes the CPU reference, launches the kernel, and
// checks agreement.
bool runTestCase(int width, int height) {
    size_t nCount = static_cast<size_t>(width) * height;
    size_t fCount = static_cast<size_t>(2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1);

    std::vector<float> N_h(nCount), P_ref(nCount), P_h(nCount);
    std::vector<float> F_h(fCount);

    for (size_t i = 0; i < nCount; ++i) {
        N_h[i] = static_cast<float>(i % 13) * 0.1f - 0.6f;
    }
    for (size_t i = 0; i < fCount; ++i) {
        F_h[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.1f;
    }

    convolution2D_h(N_h.data(), F_h.data(), P_ref.data(), width, height);
    float ms = runConstMemConvolution(N_h.data(), F_h.data(), P_h.data(), width, height);

    bool ok = true;
    for (size_t i = 0; i < nCount; ++i) {
        if (!nearlyEqual(P_h[i], P_ref[i])) {
            ok = false;
            fprintf(stderr, "Mismatch at width=%d height=%d i=%zu: gpu=%f cpu=%f\n", width, height, i, P_h[i],
                    P_ref[i]);
            break;
        }
    }

    printf("width=%-4d height=%-4d (FILTER_RADIUS=%d): %.3f ms  [%s]\n", width, height, FILTER_RADIUS, ms,
           ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    bool ok = true;
    // Mix of sizes that are and aren't multiples of BLOCK_DIM (16), so
    // ghost-cell handling on all four edges is exercised.
    ok = runTestCase(67, 51) && ok;
    ok = runTestCase(33, 33) && ok;
    ok = runTestCase(300, 200) && ok;
    ok = runTestCase(1024, 1024) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
