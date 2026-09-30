// Chapter 12: Filter
// §12.8  Related patterns -- removing duplicate keys from a sorted list
//
// §12.8 names this as a direct variant of the stable filter pattern, not a
// new algorithm: "One similar pattern to stable filter is removing duplicate
// keys from a sorted list. This pattern can be viewed as a special case of
// the filter pattern where the condition for the key to be preserved is that
// the key is not equal to the preceding key." No code figure is given (the
// section is a short conceptual survey), but the predicate is fully
// specified, and it drops directly into file 04's exact stable-filter
// machinery -- grid-wide exclusive scan of a 0/1 "keep" flag via Ch. 11
// §11.9 single-lookback -- with the ONLY change being what "keep" means:
//   file 04:  keep[i] = cond(input[i])
//   this file: keep[i] = (i == 0) || (input[i] != input[i-1])
// Every line of scan/lookback machinery below is identical to file 04's;
// only `filterKernel`'s keep computation and the CPU reference/predicate
// differ, plus an input generator that produces a SORTED array with
// genuine duplicate runs (the pattern is only meaningful on sorted input --
// §12.8's own framing).

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda/atomic>

#include "../../common/cuda_utils.h"

#define WARP_SIZE 32
#define BLOCK_DIM 256

__device__ unsigned int warpIdx() { return threadIdx.x / WARP_SIZE; }
__device__ unsigned int laneIdx() { return threadIdx.x % WARP_SIZE; }

// Inclusive warp scan of a 0/1 "keep" value (Ch. 11 §11.4 warpScan).
__device__ unsigned int warpScan(unsigned int val) {
    unsigned int lane = laneIdx();
    for (unsigned int stride = 1; stride < WARP_SIZE; stride *= 2) {
        unsigned int temp = __shfl_up_sync(0xffffffff, val, stride);
        if (lane >= stride) {
            val += temp;
        }
    }
    return val;
}

// Inclusive block-wide scan (Ch. 11 §11.4 blockScan).
__device__ unsigned int blockScan(unsigned int val, unsigned int *warpSums_s) {
    unsigned int lane = laneIdx();
    unsigned int warp = warpIdx();
    unsigned int numWarps = blockDim.x / WARP_SIZE;

    val = warpScan(val);
    if (lane == WARP_SIZE - 1) {
        warpSums_s[warp] = val;
    }
    __syncthreads();

    if (warp == 0) {
        unsigned int warpSumVal = (lane < numWarps) ? warpSums_s[lane] : 0u;
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

// Grid-wide inter-block scan via single lookback (Ch. 11 §11.9, Fig. 11.17).
__device__ unsigned int interBlockScan(unsigned int val, unsigned int bid,
                                        unsigned int *partialSums, unsigned int *flags) {
    __shared__ unsigned int previousSum;

    if (threadIdx.x == blockDim.x - 1) {
        if (bid == 0) {
            previousSum = 0u;
        } else {
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> flagRef(flags[bid - 1]);
            while (flagRef.fetch_add(0u, cuda::memory_order_acquire) == 0u) {
                // spin until the preceding block publishes its partial sum
            }
            previousSum = partialSums[bid - 1];
        }
        partialSums[bid] = previousSum + val;

        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> myFlagRef(flags[bid]);
        myFlagRef.fetch_add(1u, cuda::memory_order_release);
    }
    __syncthreads();

    return previousSum;
}

// ---------------------------------------------------------------------------
// §12.8: stable filter specialized to duplicate removal. Identical
// scan/lookback structure to file 04's filterKernel; only the keep
// computation (line marked below) differs.
// ---------------------------------------------------------------------------
__global__ void filterKernel(const unsigned int *input, unsigned int *output, unsigned int N,
                              unsigned int *outputSize, unsigned int *partialSums,
                              unsigned int *flags, unsigned int *blockCounter) {
    __shared__ unsigned int bid_s;
    if (threadIdx.x == 0) {
        bid_s = atomicAdd(blockCounter, 1u);
    }
    __syncthreads();
    unsigned int bid = bid_s;

    __shared__ unsigned int warpSums_s[BLOCK_DIM / WARP_SIZE];

    unsigned int i = bid * blockDim.x + threadIdx.x;
    unsigned int val = (i < N) ? input[i] : 0u;
    // §12.8's predicate: keep iff this is the first element, or it differs
    // from its immediate predecessor in the (sorted) input.
    unsigned int keep = (i < N && (i == 0u || val != input[i - 1])) ? 1u : 0u;

    unsigned int inclusiveLocal = blockScan(keep, warpSums_s);
    unsigned int previousBlockSum = interBlockScan(inclusiveLocal, bid, partialSums, flags);
    unsigned int offset = previousBlockSum + (inclusiveLocal - keep);

    if (keep) {
        output[offset] = val;
    }
    if (i == N - 1) {
        *outputSize = offset + keep;
    }
}

// Sorted input with genuine duplicate runs (1-6 repeats per distinct value,
// small increasing steps between distinct values) -- the pattern is only
// meaningful on sorted data, per §12.8's own framing.
std::vector<unsigned int> generateInput(unsigned int n) {
    std::vector<unsigned int> v(n);
    unsigned int state = 13579246u;
    unsigned int current = 0u;
    size_t i = 0;
    while (i < n) {
        state = state * 1103515245u + 12345u;
        unsigned int runLen = 1u + ((state >> 16) % 6u);
        for (unsigned int k = 0; k < runLen && i < n; ++k, ++i) {
            v[i] = current;
        }
        state = state * 1103515245u + 12345u;
        current += 1u + ((state >> 8) % 3u);  // strictly increasing between runs
    }
    return v;
}

// CPU reference: sequential dedupe, keeping the first occurrence of each run
// of equal values, in order.
std::vector<unsigned int> dedupeCPU(const std::vector<unsigned int> &input) {
    std::vector<unsigned int> out;
    out.reserve(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        if (i == 0 || input[i] != input[i - 1]) {
            out.push_back(input[i]);
        }
    }
    return out;
}

float runFilter(const std::vector<unsigned int> &input_h, std::vector<unsigned int> &output_h,
                 unsigned int &outputSize_h) {
    unsigned int n = static_cast<unsigned int>(input_h.size());
    size_t bytes = n * sizeof(unsigned int);
    unsigned int numBlocks = (n + BLOCK_DIM - 1) / BLOCK_DIM;

    unsigned int *input_d, *output_d, *outputSize_d, *partialSums_d;
    unsigned int *flags_d, *blockCounter_d;
    CUDA_CHECK(cudaMalloc((void **)&input_d, bytes));
    CUDA_CHECK(cudaMalloc((void **)&output_d, bytes));
    CUDA_CHECK(cudaMalloc((void **)&outputSize_d, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc((void **)&partialSums_d, numBlocks * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc((void **)&flags_d, numBlocks * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc((void **)&blockCounter_d, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(input_d, input_h.data(), bytes, cudaMemcpyHostToDevice));

    dim3 dimBlock(BLOCK_DIM);
    dim3 dimGrid(numBlocks);

    auto resetState = [&]() {
        CUDA_CHECK(cudaMemset(flags_d, 0, numBlocks * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(blockCounter_d, 0, sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(outputSize_d, 0, sizeof(unsigned int)));
    };

    resetState();
    filterKernel<<<dimGrid, dimBlock>>>(input_d, output_d, n, outputSize_d, partialSums_d,
                                        flags_d, blockCounter_d);  // warm-up
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    resetState();
    GpuTimer timer;
    timer.start();
    filterKernel<<<dimGrid, dimBlock>>>(input_d, output_d, n, outputSize_d, partialSums_d,
                                        flags_d, blockCounter_d);
    CUDA_CHECK(cudaGetLastError());
    float ms = timer.stopAndGetMs();

    CUDA_CHECK(cudaMemcpy(&outputSize_h, outputSize_d, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    output_h.resize(outputSize_h);
    if (outputSize_h > 0) {
        CUDA_CHECK(cudaMemcpy(output_h.data(), output_d, outputSize_h * sizeof(unsigned int),
                               cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaFree(input_d));
    CUDA_CHECK(cudaFree(output_d));
    CUDA_CHECK(cudaFree(outputSize_d));
    CUDA_CHECK(cudaFree(partialSums_d));
    CUDA_CHECK(cudaFree(flags_d));
    CUDA_CHECK(cudaFree(blockCounter_d));
    return ms;
}

bool runTestCase(unsigned int n) {
    std::vector<unsigned int> input_h = generateInput(n);
    std::vector<unsigned int> ref = dedupeCPU(input_h);

    std::vector<unsigned int> gpu;
    unsigned int outputSize = 0;
    float ms = runFilter(input_h, gpu, outputSize);

    bool ok = (outputSize == ref.size()) && (gpu == ref);
    printf("N=%u (blocks=%u): cpu unique=%zu gpu unique=%u  %.4f ms  [%s]\n",
           n, (n + BLOCK_DIM - 1) / BLOCK_DIM, ref.size(), outputSize, ms, ok ? "match" : "MISMATCH");
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
