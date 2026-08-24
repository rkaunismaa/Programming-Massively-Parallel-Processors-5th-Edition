// Chapter 11: Scan
// §11.7  Register tiling to avoid shared memory access latency (Fig. 11.13)
//
// File 04's coarsened kernel stores each thread's COARSE_FACTOR-element
// subsegment in shared memory and repeatedly reads/writes it there during
// the thread's own sequential scan -- wasteful, since that subsegment is
// private to the thread and never read by any other thread. §11.7's fix:
// load the subsegment from shared memory into a small local (register)
// array ONCE, do the sequential scan entirely in registers, and write the
// finished result back to shared memory once (so the block-level add step
// and the final coalesced store can still see it).
//
// Shared memory is still used as an intermediary for the initial load and
// final store -- not because the data is reused there, but purely to keep
// those global-memory transfers coalesced (loading/storing a thread's
// register tile directly to/from global memory would touch
// COARSE_FACTOR-strided addresses per thread, exactly like file 04's
// callout). §11.7 draws the explicit parallel to Chapter 6's corner-turning
// optimization for this reason.
//
// `#pragma unroll` on the register-array loops is a reminder to the reader
// (per the book) that these loops need constant indices to let the compiler
// promote `buffer_r` to registers -- the compiler usually unrolls small
// fixed-trip-count loops on its own, but the annotation documents intent.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#include "../../common/cuda_utils.h"

#define WARP_SIZE 32
#define COARSE_FACTOR 4
#define BLOCK_DIM 256

__device__ unsigned int warpIdx() { return threadIdx.x / WARP_SIZE; }
__device__ unsigned int laneIdx() { return threadIdx.x % WARP_SIZE; }

__device__ float warpScan(float val) {
    unsigned int lane = laneIdx();
    for (unsigned int stride = 1; stride < WARP_SIZE; stride *= 2) {
        float temp = __shfl_up_sync(0xffffffff, val, stride);
        if (lane >= stride) {
            val += temp;
        }
    }
    return val;
}

__device__ float blockScan(float val, float *warpSums_s) {
    unsigned int lane = laneIdx();
    unsigned int warp = warpIdx();
    unsigned int numWarps = blockDim.x / WARP_SIZE;

    val = warpScan(val);
    if (lane == WARP_SIZE - 1) {
        warpSums_s[warp] = val;
    }
    __syncthreads();

    if (warp == 0) {
        float warpSumVal = (lane < numWarps) ? warpSums_s[lane] : 0.0f;
        warpSumVal = warpScan(warpSumVal);
        if (lane < numWarps) {
            warpSums_s[lane] = warpSumVal;
        }
    }
    __syncthreads();

    if (warp > 0) {
        val += warpSums_s[warp - 1];
    }
    return val;
}

// ---------------------------------------------------------------------------
// §11.7, Fig. 11.13: register-tiled scan-scan-add kernel. Structurally the
// same as file 04's coarsenedScanKernel, but the per-thread sequential scan
// runs on a local array `buffer_r` instead of repeatedly touching shared
// memory.
// ---------------------------------------------------------------------------
__global__ void registerTiledScanKernel(const float *input, float *output, unsigned int N) {
    __shared__ float buffer_s[COARSE_FACTOR * BLOCK_DIM];
    __shared__ float warpSums_s[BLOCK_DIM / WARP_SIZE];
    __shared__ float scannedThreadSums_s[BLOCK_DIM];

    unsigned int segment = blockIdx.x * blockDim.x * COARSE_FACTOR;

    // Coalesced load into shared memory (purely to enable coalescing --
    // see file header).
    for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        unsigned int idx = segment + c * BLOCK_DIM + threadIdx.x;
        buffer_s[c * BLOCK_DIM + threadIdx.x] = (idx < N) ? input[idx] : 0.0f;
    }
    __syncthreads();

    // Load the thread's own contiguous subsegment into registers and scan
    // it there.
    float buffer_r[COARSE_FACTOR];
    unsigned int threadSegment = threadIdx.x * COARSE_FACTOR;
    buffer_r[0] = buffer_s[threadSegment];
#pragma unroll
    for (unsigned int c = 1; c < COARSE_FACTOR; ++c) {
        buffer_r[c] = buffer_r[c - 1] + buffer_s[threadSegment + c];
    }

    // Block-level scan of the per-thread subsegment sums.
    float threadSum = buffer_r[COARSE_FACTOR - 1];
    float scannedThreadSum = blockScan(threadSum, warpSums_s);
    scannedThreadSums_s[threadIdx.x] = scannedThreadSum;
    __syncthreads();

    // Add the sum of all preceding threads' elements (still in registers),
    // then return the finished subsegment to shared memory.
    float prev = (threadIdx.x > 0) ? scannedThreadSums_s[threadIdx.x - 1] : 0.0f;
#pragma unroll
    for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        buffer_s[threadSegment + c] = buffer_r[c] + prev;
    }
    __syncthreads();

    // Coalesced store.
    for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        unsigned int idx = segment + c * BLOCK_DIM + threadIdx.x;
        if (idx < N) {
            output[idx] = buffer_s[c * BLOCK_DIM + threadIdx.x];
        }
    }
}

void scanSegmentsCPU(const float *input, float *output, unsigned int N, unsigned int segSize) {
    for (unsigned int base = 0; base < N; base += segSize) {
        unsigned int end = std::min(base + segSize, N);
        float acc = 0.0f;
        for (unsigned int i = base; i < end; ++i) {
            acc += input[i];
            output[i] = acc;
        }
    }
}

std::vector<float> generateInput(unsigned int n) {
    std::vector<float> v(n);
    unsigned int state = 523456789u;
    for (unsigned int i = 0; i < n; ++i) {
        state = state * 1103515245u + 12345u;
        v[i] = static_cast<float>((state >> 8) & 0xFFFFFF) / static_cast<float>(0x1000000);
    }
    return v;
}

float runRegisterTiledScan(const std::vector<float> &input_h, std::vector<float> &output_h,
                            unsigned int numBlocks) {
    unsigned int n = static_cast<unsigned int>(input_h.size());
    size_t bytes = n * sizeof(float);

    float *input_d, *output_d;
    CUDA_CHECK(cudaMalloc((void **)&input_d, bytes));
    CUDA_CHECK(cudaMalloc((void **)&output_d, bytes));
    CUDA_CHECK(cudaMemcpy(input_d, input_h.data(), bytes, cudaMemcpyHostToDevice));

    dim3 dimBlock(BLOCK_DIM);
    dim3 dimGrid(numBlocks);

    registerTiledScanKernel<<<dimGrid, dimBlock>>>(input_d, output_d, n);  // warm-up
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    registerTiledScanKernel<<<dimGrid, dimBlock>>>(input_d, output_d, n);
    CUDA_CHECK(cudaGetLastError());
    float ms = timer.stopAndGetMs();

    output_h.resize(n);
    CUDA_CHECK(cudaMemcpy(output_h.data(), output_d, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(input_d));
    CUDA_CHECK(cudaFree(output_d));
    return ms;
}

bool runTestCase(unsigned int numBlocks) {
    unsigned int segSize = COARSE_FACTOR * BLOCK_DIM;
    unsigned int n = segSize * numBlocks;
    std::vector<float> input_h = generateInput(n);

    std::vector<float> ref(n);
    scanSegmentsCPU(input_h.data(), ref.data(), n, segSize);

    std::vector<float> gpu;
    float ms = runRegisterTiledScan(input_h, gpu, numBlocks);

    bool ok = true;
    for (unsigned int i = 0; i < n; ++i) {
        if (!nearlyEqual(gpu[i], ref[i], 1e-2f)) {
            ok = false;
            printf("  mismatch at %u: cpu=%.6f gpu=%.6f\n", i, ref[i], gpu[i]);
            break;
        }
    }
    printf("blocks=%u N=%u (COARSE_FACTOR=%d, BLOCK_DIM=%d): last=%.6f (cpu) / %.6f (gpu)  %.4f ms  [%s]\n",
           numBlocks, n, COARSE_FACTOR, BLOCK_DIM, ref[n - 1], gpu[n - 1], ms, ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    bool ok = true;
    ok = runTestCase(1) && ok;
    ok = runTestCase(4) && ok;
    ok = runTestCase(16) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
