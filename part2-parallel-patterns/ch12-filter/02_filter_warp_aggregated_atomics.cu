// Chapter 12: Filter
// §12.3  Coalescing atomic operations with warp-level primitives
//        (Figs. 12.3 - 12.4)
//
// File 01's atomic fetch_add() on the shared outputSize counter is a
// bottleneck: every cond()-passing thread in a warp issues its OWN atomic
// on the SAME address, guaranteeing a hardware-serialized pileup exactly
// when it hurts most (threads in a warp execute in lockstep, so their
// atomics are issued at the same time). §12.3's fix -- a "coalesced atomic
// operation" -- has the warp elect one leader thread to issue a SINGLE
// atomic on behalf of every active thread, then hands each thread its own
// slot by shuffling the base index back out and adding a per-thread rank.
//
// The book gives two versions of the identical idea and this file
// implements both, verified independently against the same CPU reference:
//
//   - filterKernelIntrinsics (Fig. 12.3): built from raw warp-voting/shuffle
//     primitives --
//       __activemask()   -- which lanes are active at the atomic site
//       __ffs(mask)-1    -- index of the lowest active lane -> leader
//       __popc(mask)     -- population count -> total slots this warp needs
//       __shfl_sync(mask, j, leader) -- leader's base index broadcast to
//                                       every active lane
//       __popc(mask & ((1<<lane)-1)) -- count of active lanes BEFORE this
//                                       one -> this thread's own offset
//     This is a binary prefix sum on the active-mask bits (Harris & Garland
//     [1], cited directly in §12.3), computed with bitwise ops instead of a
//     real scan.
//   - filterKernelCoopGroups (Fig. 12.4): the same four steps, rewritten
//     against the cooperative_groups API's coalesced_group, which packages
//     the mask/leader/popcount/shuffle bookkeeping: coalesced_threads()
//     replaces __activemask(), thread_rank()==0 replaces __ffs()-1,
//     group.size() replaces __popc(), group.shfl(j,0) replaces
//     __shfl_sync(), and thread_rank() itself IS the intra-warp offset
//     (replacing the explicit previousActiveThreads/__popc bit-mask dance).
//
// Both kernels only coalesce the atomic among the threads of ONE warp that
// are active at the SAME point in the SAME instruction stream -- the
// compaction is still unstable, exactly like file 01, since the order in
// which different warps' fetch_add() calls land is still arbitrary.
//
// §12.3 also notes explicitly that nvcc already performs this optimization
// automatically in practice, and skips it entirely when only one thread in
// the warp is active -- so this file makes no speed claim; it exists to
// demonstrate the warp-voting/shuffle mechanics the book introduces, with
// both variants checked for correctness against each other and the CPU.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda/atomic>
#include <cooperative_groups.h>

#include "../../common/cuda_utils.h"

namespace cg = cooperative_groups;

#define BLOCK_DIM 256
#define WARP_SIZE 32

__device__ __host__ bool cond(unsigned int val) { return (val % 2u) == 0u; }

// ---------------------------------------------------------------------------
// §12.3, Fig. 12.3: coalesced atomic via raw warp-voting intrinsics.
// ---------------------------------------------------------------------------
__global__ void filterKernelIntrinsics(const unsigned int *input, unsigned int *output,
                                        unsigned int N, unsigned int *outputSize) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        unsigned int val = input[i];
        if (cond(val)) {
            unsigned int activeThreads = __activemask();
            unsigned int j;
            // Assign a leader thread: lowest-index active lane.
            unsigned int leader = __ffs(activeThreads) - 1;
            unsigned int laneIdx = threadIdx.x % WARP_SIZE;
            if (laneIdx == leader) {
                // Find how many threads are active.
                unsigned int numActive = __popc(activeThreads);
                // Have the leader perform the atomic operation.
                cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
                    outputSize_ref(*outputSize);
                j = outputSize_ref.fetch_add(numActive, cuda::memory_order_relaxed);
            }
            // Broadcast result to other threads.
            j = __shfl_sync(activeThreads, j, leader);
            // Find the position of each active thread in the output.
            unsigned int previousThreads = (1u << laneIdx) - 1u;
            unsigned int previousActiveThreads = activeThreads & previousThreads;
            unsigned int offset = __popc(previousActiveThreads);
            // Store the result.
            output[j + offset] = val;
        }
    }
}

// ---------------------------------------------------------------------------
// §12.3, Fig. 12.4: the same coalesced atomic, expressed via the
// cooperative_groups API's coalesced_group.
// ---------------------------------------------------------------------------
__global__ void filterKernelCoopGroups(const unsigned int *input, unsigned int *output,
                                        unsigned int N, unsigned int *outputSize) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        unsigned int val = input[i];
        if (cond(val)) {
            cg::coalesced_group activeThreads = cg::coalesced_threads();
            unsigned int j;
            // Assign a leader thread: rank 0 in the coalesced group.
            if (activeThreads.thread_rank() == 0) {
                // Find how many threads are active.
                unsigned int numActive = activeThreads.size();
                // Have the leader perform the atomic operation.
                cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
                    outputSize_ref(*outputSize);
                j = outputSize_ref.fetch_add(numActive, cuda::memory_order_relaxed);
            }
            // Broadcast result to other threads.
            j = activeThreads.shfl(j, 0);
            // Find the position of each active thread in the output: the
            // thread's own rank in the coalesced group.
            unsigned int offset = activeThreads.thread_rank();
            // Store the result.
            output[j + offset] = val;
        }
    }
}

std::vector<unsigned int> generateInput(unsigned int n) {
    std::vector<unsigned int> v(n);
    unsigned int state = 223456789u;
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

template <typename KernelFn>
float runFilter(KernelFn kernel, const std::vector<unsigned int> &input_h,
                 std::vector<unsigned int> &output_h, unsigned int &outputSize_h) {
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
    kernel<<<dimGrid, dimBlock>>>(input_d, output_d, n, outputSize_d);  // warm-up
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemset(outputSize_d, 0, sizeof(unsigned int)));
    GpuTimer timer;
    timer.start();
    kernel<<<dimGrid, dimBlock>>>(input_d, output_d, n, outputSize_d);
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

bool checkSetMatch(const std::vector<unsigned int> &gpu, const std::vector<unsigned int> &ref) {
    if (gpu.size() != ref.size()) return false;
    std::vector<unsigned int> gpuSorted = gpu, refSorted = ref;
    std::sort(gpuSorted.begin(), gpuSorted.end());
    std::sort(refSorted.begin(), refSorted.end());
    return gpuSorted == refSorted;
}

bool runTestCase(unsigned int n) {
    std::vector<unsigned int> input_h = generateInput(n);
    std::vector<unsigned int> ref = filterCPU(input_h);

    std::vector<unsigned int> gpuIntr, gpuCg;
    unsigned int sizeIntr = 0, sizeCg = 0;
    float msIntr = runFilter(filterKernelIntrinsics, input_h, gpuIntr, sizeIntr);
    float msCg = runFilter(filterKernelCoopGroups, input_h, gpuCg, sizeCg);

    bool okIntr = checkSetMatch(gpuIntr, ref);
    bool okCg = checkSetMatch(gpuCg, ref);

    printf("N=%u: cpu kept=%zu | intrinsics kept=%u %.4f ms [%s] | coop_groups kept=%u %.4f ms [%s]\n",
           n, ref.size(), sizeIntr, msIntr, okIntr ? "match" : "MISMATCH",
           sizeCg, msCg, okCg ? "match" : "MISMATCH");
    return okIntr && okCg;
}

int main() {
    bool ok = true;
    ok = runTestCase(1024) && ok;
    ok = runTestCase(100000) && ok;
    ok = runTestCase(1 << 20) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
