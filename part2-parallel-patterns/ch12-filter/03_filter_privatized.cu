// Chapter 12: Filter
// §12.4  Privatization (Figs. 12.5 - 12.6)
//
// A second, independent way to cut contention on the global outputSize
// counter (independent of file 02's warp-coalesced atomics, and applicable
// together with it in principle): give every thread BLOCK a private copy of
// the counter and the output list in shared memory. Within a block, threads
// still atomically race on this private counter -- but shared-memory
// atomics are far cheaper than a device-wide atomic, and contention is now
// scoped to one block's worth of threads instead of the whole grid. Only
// once, at the very end, does a single thread per block reserve one
// contiguous chunk of the PUBLIC output array (one atomic per block, not
// one per surviving key) and the block's threads copy their private output
// list into it with consecutive threads writing consecutive slots --
// picking up the same memory-coalescing bonus files 02 and 03 both get from
// avoiding a scattered one-atomic-per-key pattern.
//
// Fig. 12.6 transcribed directly:
//   1. output_s[BLOCK_DIM] / outputSize_s: block-private list + counter in
//      shared memory, counter zero-initialized by thread 0 (lines 04-10).
//   2. Every thread filters into the PRIVATE list via a
//      cuda::thread_scope_block atomic (the counter in shared memory is
//      only ever touched by threads of this one block -- lines 13-23).
//   3. Thread 0 reserves space in the PUBLIC list with one
//      cuda::thread_scope_device atomic, incrementing it by the block's
//      total private count in one shot (lines 26-32).
//   4. Threads collaboratively copy output_s -> output[j + threadIdx.x],
//      consecutive threads writing consecutive global addresses (lines
//      34-37).
//
// This file's filter condition and unstable-output-set check follow file
// 01/02's convention exactly -- privatization changes nothing about
// ordering, only about how contention on the counter is reduced.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda/atomic>

#include "../../common/cuda_utils.h"

#define BLOCK_DIM 256

__device__ __host__ bool cond(unsigned int val) { return (val % 2u) == 0u; }

// ---------------------------------------------------------------------------
// §12.4, Fig. 12.6: unstable filter kernel with privatization.
// ---------------------------------------------------------------------------
__global__ void filterKernel(const unsigned int *input, unsigned int *output,
                              unsigned int N, unsigned int *outputSize) {
    // Declare and initialize private output list.
    __shared__ unsigned int output_s[BLOCK_DIM];
    __shared__ unsigned int outputSize_s;
    if (threadIdx.x == 0) {
        outputSize_s = 0;
    }
    __syncthreads();

    // Filter in the private lists.
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        unsigned int val = input[i];
        if (cond(val)) {
            cuda::atomic_ref<unsigned int, cuda::thread_scope_block>
                outputSize_s_ref(outputSize_s);
            unsigned int j = outputSize_s_ref.fetch_add(1, cuda::memory_order_relaxed);
            output_s[j] = val;
        }
    }
    __syncthreads();

    // Update the public counter.
    __shared__ unsigned int j;
    if (threadIdx.x == 0) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
            outputSize_ref(*outputSize);
        j = outputSize_ref.fetch_add(outputSize_s, cuda::memory_order_relaxed);
    }
    __syncthreads();

    // Write to the public list.
    if (threadIdx.x < outputSize_s) {
        output[j + threadIdx.x] = output_s[threadIdx.x];
    }
}

std::vector<unsigned int> generateInput(unsigned int n) {
    std::vector<unsigned int> v(n);
    unsigned int state = 323456789u;
    for (unsigned int i = 0; i < n; ++i) {
        state = state * 1103515245u + 12345u;
        v[i] = (state >> 8) & 0xFFFFu;
    }
    return v;
}

std::vector<unsigned int> filterCPU(const std::vector<unsigned int> &input) {
    std::vector<unsigned int> out;
    out.reserve(input.size());
    for (unsigned int v : input) {
        if (cond(v)) out.push_back(v);
    }
    return out;
}

float runFilter(const std::vector<unsigned int> &input_h, std::vector<unsigned int> &output_h,
                 unsigned int &outputSize_h) {
    unsigned int n = static_cast<unsigned int>(input_h.size());
    size_t bytes = n * sizeof(unsigned int);

    unsigned int *input_d, *output_d, *outputSize_d;
    CUDA_CHECK(cudaMalloc((void **)&input_d, bytes));
    CUDA_CHECK(cudaMalloc((void **)&output_d, bytes));
    CUDA_CHECK(cudaMalloc((void **)&outputSize_d, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(input_d, input_h.data(), bytes, cudaMemcpyHostToDevice));

    dim3 dimBlock(BLOCK_DIM);
    dim3 dimGrid((n + BLOCK_DIM - 1) / BLOCK_DIM);

    CUDA_CHECK(cudaMemset(outputSize_d, 0, sizeof(unsigned int)));
    filterKernel<<<dimGrid, dimBlock>>>(input_d, output_d, n, outputSize_d);  // warm-up
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemset(outputSize_d, 0, sizeof(unsigned int)));
    GpuTimer timer;
    timer.start();
    filterKernel<<<dimGrid, dimBlock>>>(input_d, output_d, n, outputSize_d);
    CUDA_CHECK(cudaGetLastError());
    float ms = timer.stopAndGetMs();

    CUDA_CHECK(cudaMemcpy(&outputSize_h, outputSize_d, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    output_h.resize(outputSize_h);
    CUDA_CHECK(cudaMemcpy(output_h.data(), output_d, outputSize_h * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(input_d));
    CUDA_CHECK(cudaFree(output_d));
    CUDA_CHECK(cudaFree(outputSize_d));
    return ms;
}

bool runTestCase(unsigned int n) {
    std::vector<unsigned int> input_h = generateInput(n);
    std::vector<unsigned int> ref = filterCPU(input_h);

    std::vector<unsigned int> gpu;
    unsigned int outputSize = 0;
    float ms = runFilter(input_h, gpu, outputSize);

    bool ok = (outputSize == ref.size());
    if (ok) {
        std::vector<unsigned int> gpuSorted = gpu, refSorted = ref;
        std::sort(gpuSorted.begin(), gpuSorted.end());
        std::sort(refSorted.begin(), refSorted.end());
        ok = (gpuSorted == refSorted);
    }
    printf("N=%u: cpu kept=%zu gpu kept=%u  %.4f ms  [%s]\n",
           n, ref.size(), outputSize, ms, ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    bool ok = true;
    ok = runTestCase(1024) && ok;
    ok = runTestCase(100000) && ok;
    ok = runTestCase(1 << 20) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
