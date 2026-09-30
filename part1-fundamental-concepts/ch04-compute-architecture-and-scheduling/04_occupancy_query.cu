// Chapter 4: Compute architecture and scheduling
// §4.7  Resource partitioning and occupancy -- the CUDA Occupancy API
//
// §4.7 defines occupancy as "the ratio of the number of warps assigned to
// an SM to the maximum number of warps it supports" and walks through
// several fully worked numeric examples of how block size and per-thread
// register usage each limit it (a Hopper H100 has 2048 thread slots, 64
// warp slots, 32 block slots, and 65,536 registers per SM in those
// examples). It then closes the topic: "The reader is referred to the
// Occupancy Calculator... The application host code can also call the
// CUDA Occupancy API functions such as
// cudaOccupancyMaxActiveBlocksPerMultiprocessor() to determine the
// occupancy of a kernel when called with a grid configuration on the GPU
// being used." That API call is named in the text but never exercised by
// any sample in this chapter -- this sample exercises it.
//
// Rather than hardcode the book's H100 numbers (this repo has no H100 to
// verify them against), this sample queries the *actual* GPU's limits via
// cudaGetDeviceProperties() (as in 02_query_device_properties.cu) and asks
// the real occupancy API for the real answer, for two kernels with
// deliberately different register footprints:
//   - lowRegKernel:  one multiply per thread -- minimal register use.
//   - highRegKernel: 32 independent live temporaries combined pairwise, so
//     the compiler cannot fold them away -- deliberately higher register
//     use, to reproduce the section's "some kernels require many registers
//     per thread, and some require few" observation and its consequence
//     for occupancy.
//
// For a range of block sizes, this prints each kernel's actual per-thread
// register count (via cudaFuncGetAttributes(), the numRegs field the
// section's discussion of devProp.regsPerBlock is adjacent to) and the
// occupancy cudaOccupancyMaxActiveBlocksPerMultiprocessor() reports for
// it -- reproducing, with real hardware numbers instead of the book's H100
// figures, both of §4.7's qualitative lessons: (1) occupancy varies with
// block size for a single kernel, and (2) a higher-register-per-thread
// kernel can never achieve more resident blocks per SM, at a given block
// size, than a lower-register-per-thread one.
//
// PASS requires: every device-property and occupancy API call succeeds;
// both kernels' outputs match a CPU reference; and, at every block size
// tested, highRegKernel's active-blocks-per-SM never exceeds
// lowRegKernel's (the register-pressure sanity check above). Occupancy
// numbers themselves are printed for information -- they are a property of
// the specific GPU this runs on, not a pass/fail criterion.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

constexpr int UNROLL = 32;

// ---------------------------------------------------------------------------
// Minimal per-thread register footprint: one live value, one multiply.
// ---------------------------------------------------------------------------
__global__ void lowRegKernel(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = in[i] * 2.0f;
    }
}

// ---------------------------------------------------------------------------
// Deliberately higher per-thread register footprint: UNROLL independent
// temporaries, all simultaneously live (each depends on the input, and
// each contributes to the final sum), so the compiler cannot reduce this
// to a handful of registers the way it could a simple reduction.
// ---------------------------------------------------------------------------
__global__ void highRegKernel(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = in[i];
        float r[UNROLL];
#pragma unroll
        for (int j = 0; j < UNROLL; ++j) {
            r[j] = v * (1.0f + 0.001f * static_cast<float>(j));
        }
        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < UNROLL; ++j) {
            acc += r[j] * r[(j + 1) % UNROLL];
        }
        out[i] = acc;
    }
}

void referenceLow(const float *in, float *out, int n) {
    for (int i = 0; i < n; ++i) {
        out[i] = in[i] * 2.0f;
    }
}

void referenceHigh(const float *in, float *out, int n) {
    for (int i = 0; i < n; ++i) {
        float v = in[i];
        float r[UNROLL];
        for (int j = 0; j < UNROLL; ++j) {
            r[j] = v * (1.0f + 0.001f * static_cast<float>(j));
        }
        float acc = 0.0f;
        for (int j = 0; j < UNROLL; ++j) {
            acc += r[j] * r[(j + 1) % UNROLL];
        }
        out[i] = acc;
    }
}

int main() {
    bool ok = true;

    cudaDeviceProp devProp;
    CUDA_CHECK(cudaGetDeviceProperties(&devProp, 0));
    printf("Device 0: %s (compute capability %d.%d)\n", devProp.name, devProp.major,
           devProp.minor);
    printf("  multiProcessorCount           : %d\n", devProp.multiProcessorCount);
    printf("  maxThreadsPerMultiProcessor    : %d\n", devProp.maxThreadsPerMultiProcessor);
    printf("  maxThreadsPerBlock             : %d\n", devProp.maxThreadsPerBlock);
    printf("  regsPerMultiprocessor          : %d\n", devProp.regsPerMultiprocessor);
    printf("\n");

    cudaFuncAttributes attrLow, attrHigh;
    CUDA_CHECK(cudaFuncGetAttributes(&attrLow, lowRegKernel));
    CUDA_CHECK(cudaFuncGetAttributes(&attrHigh, highRegKernel));
    printf("lowRegKernel  numRegs (registers/thread) : %d\n", attrLow.numRegs);
    printf("highRegKernel numRegs (registers/thread) : %d\n", attrHigh.numRegs);
    printf("\n");

    const int candidateBlockSizes[] = {32, 64, 128, 256, 512, 1024};
    printf("%10s %14s %10s %14s %10s\n", "blockSize", "low activeBlk", "low occ%",
           "high activeBlk", "high occ%");
    for (int blockSize : candidateBlockSizes) {
        if (blockSize > devProp.maxThreadsPerBlock) {
            continue;
        }

        int activeLow = 0, activeHigh = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeLow, lowRegKernel,
                                                                   blockSize, 0));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeHigh, highRegKernel,
                                                                   blockSize, 0));

        double occLow =
            100.0 * activeLow * blockSize / static_cast<double>(devProp.maxThreadsPerMultiProcessor);
        double occHigh = 100.0 * activeHigh * blockSize /
                          static_cast<double>(devProp.maxThreadsPerMultiProcessor);

        printf("%10d %14d %9.1f%% %14d %9.1f%%\n", blockSize, activeLow, occLow, activeHigh,
               occHigh);

        // §4.7's register-pressure lesson, checked against the real API's
        // answer rather than asserted by hand: a kernel needing more
        // registers per thread can never fit more resident blocks per SM,
        // at the same block size, than one needing fewer.
        if (activeHigh > activeLow) {
            ok = false;
            fprintf(stderr,
                    "Occupancy sanity check failed at blockSize=%d: highReg fit more blocks "
                    "(%d) than lowReg (%d)\n",
                    blockSize, activeHigh, activeLow);
        }
    }
    printf("\n");

    // Numeric correctness, on a modest array (occupancy analysis above
    // already covers the full array size range this kernel would run at).
    const int n = 1 << 16;
    std::vector<float> in_h(n);
    for (int i = 0; i < n; ++i) {
        in_h[i] = 1.0f + static_cast<float>(i % 97) * 0.01f;
    }

    std::vector<float> refLow_h(n), refHigh_h(n);
    referenceLow(in_h.data(), refLow_h.data(), n);
    referenceHigh(in_h.data(), refHigh_h.data(), n);

    float *in_d, *outLow_d, *outHigh_d;
    CUDA_CHECK(cudaMalloc((void **)&in_d, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&outLow_d, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&outHigh_d, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(in_d, in_h.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    const int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;
    lowRegKernel<<<gridSize, blockSize>>>(in_d, outLow_d, n);
    CUDA_CHECK(cudaGetLastError());
    highRegKernel<<<gridSize, blockSize>>>(in_d, outHigh_d, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> outLow_h(n), outHigh_h(n);
    CUDA_CHECK(cudaMemcpy(outLow_h.data(), outLow_d, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(outHigh_h.data(), outHigh_d, n * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(in_d));
    CUDA_CHECK(cudaFree(outLow_d));
    CUDA_CHECK(cudaFree(outHigh_d));

    for (int i = 0; i < n; ++i) {
        if (!nearlyEqual(outLow_h[i], refLow_h[i]) || !nearlyEqual(outHigh_h[i], refHigh_h[i])) {
            ok = false;
            fprintf(stderr, "Mismatch at %d: low=%f (ref %f) high=%f (ref %f)\n", i,
                    outLow_h[i], refLow_h[i], outHigh_h[i], refHigh_h[i]);
            break;
        }
    }

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
