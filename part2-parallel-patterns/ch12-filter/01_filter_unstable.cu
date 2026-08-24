// Chapter 12: Filter
// §12.2  A simple parallel unstable filter (Fig. 12.2)
//
// Stream compaction: remove every element that fails cond() and pack the
// keys that pass into a dense output array. This file is the book's
// starting point -- every thread that keeps its key claims a slot in the
// output by atomically incrementing a single global counter (outputSize),
// then stores its key at the returned index.
//
// Because the hardware may service the fetch_add() atomics from different
// threads in any order, a thread's place in the output array has nothing
// to do with its key's position in the input array -- hence "unstable":
// the relative order of surviving keys is not preserved. §12.2 calls out
// the resulting bottleneck explicitly: every cond()-passing thread across
// the whole grid contends on the very same counter, and the hardware
// serializes all of them. §12.3-12.4 (files 02-03) attack this contention
// two different ways; this file implements the un-optimized baseline
// exactly as Fig. 12.2 gives it.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda/atomic>

#include "../../common/cuda_utils.h"

#define BLOCK_DIM 256

// The filtering predicate cond() from Fig. 12.2 line 06. Keep even-valued
// keys -- deterministic, easy to check on the host, and keeps roughly half
// the input so every test case exercises real compaction.
__device__ __host__ bool cond(unsigned int val) { return (val % 2u) == 0u; }

// ---------------------------------------------------------------------------
// §12.2, Fig. 12.2: simple unstable filter kernel. outputSize must be
// zero-initialized by the caller before launch.
// ---------------------------------------------------------------------------
__global__ void filterKernel(const unsigned int *input, unsigned int *output,
                              unsigned int N, unsigned int *outputSize) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        unsigned int val = input[i];
        if (cond(val)) {
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
                outputSize_ref(*outputSize);
            unsigned int j = outputSize_ref.fetch_add(1, cuda::memory_order_relaxed);
            output[j] = val;
        }
    }
}

std::vector<unsigned int> generateInput(unsigned int n) {
    std::vector<unsigned int> v(n);
    unsigned int state = 123456789u;
    for (unsigned int i = 0; i < n; ++i) {
        state = state * 1103515245u + 12345u;
        v[i] = (state >> 8) & 0xFFFFu;
    }
    return v;
}

// CPU reference: sequential filter, preserving input order (this is what
// "stable" would look like -- used here only to derive the expected
// multiset of surviving keys, since file 01's GPU output order is
// unstable and must be compared as a set, not element-by-element).
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

    // Unstable filter: verify the output SET matches (sort-and-compare),
    // not element-by-element order.
    bool ok = (outputSize == ref.size());
    if (ok) {
        std::vector<unsigned int> gpuSorted = gpu;
        std::vector<unsigned int> refSorted = ref;
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
