// Chapter 12: Filter
// §12.5  A simple parallel stable filter (Figs. 12.7 - 12.8)
//
// Files 01-03 are all UNSTABLE: a thread claims whatever slot the atomic
// counter happens to hand back, so surviving keys land in the output in an
// arbitrary order. A stable filter must instead place each surviving key at
// EXACTLY the slot equal to how many earlier input keys also survived --
// i.e. the EXCLUSIVE prefix sum ("scan", Chapter 11) of a 0/1 "keep" flag
// computed across the whole grid (Fig. 12.7's worked example: keep = [0,1,
// 0,1,1,0,1,0,1,0,1,1,0,0,1,0], exclusive scan of keep = each surviving
// key's output index).
//
// Fig. 12.8's kernel is deliberately written against a black-box
// `gridExclusiveScan(keep)` -- the book explains this is "performed within
// a single kernel following the single-kernel scan implementation discussed
// in Chapter 11" (§11.9, single-lookback). This file supplies that grid-wide
// exclusive scan explicitly, reusing this project's own Chapter 11 file 06
// machinery (warpScan / blockScan / interBlockScan via dynamic block-index
// assignment + device-scope cuda::atomic_ref lookback), specialized to
// `unsigned int` "keep" values and converted from inclusive to exclusive
// (subtract the thread's own keep bit from its inclusive block-local scan).
// Structurally this file's kernel is Fig. 12.8 line-for-line:
//   03-04: load val
//   05:    keep = cond(val) ? 1 : 0
//   06:    offset = gridExclusiveScan(keep)     <- expanded below
//   07-09: if(keep) output[offset] = val
//   10-12: if(i==N-1) *outputSize = offset+keep
// with added N-boundary safety (Fig. 12.8's own listing assumes N is an
// exact multiple of the launch size and omits a bounds check).

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda/atomic>

#include "../../common/cuda_utils.h"

#define WARP_SIZE 32
#define BLOCK_DIM 256

__device__ __host__ bool cond(unsigned int val) { return (val % 2u) == 0u; }

__device__ unsigned int warpIdx() { return threadIdx.x / WARP_SIZE; }
__device__ unsigned int laneIdx() { return threadIdx.x % WARP_SIZE; }

// Inclusive warp scan of a 0/1 "keep" value (Ch. 11 §11.4 warpScan,
// specialized to unsigned int).
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

// Grid-wide inter-block scan via single lookback (Ch. 11 §11.9,
// Fig. 11.17): only the block's leader thread (last thread) walks the
// lookback chain; every thread receives the result through shared memory.
// `val` need only be correct at the leader thread -- for a non-coarsened,
// one-element-per-thread kernel, the leader's own INCLUSIVE block-local
// scan value already equals the whole block's total "keep" count.
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
// §12.5, Fig. 12.8: single-kernel stable filter via grid-wide exclusive scan.
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
    unsigned int keep = (i < N && cond(val)) ? 1u : 0u;

    // gridExclusiveScan(keep), expanded: block-local inclusive scan, then
    // fold in every preceding block's total via single lookback.
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

std::vector<unsigned int> generateInput(unsigned int n) {
    std::vector<unsigned int> v(n);
    unsigned int state = 423456789u;
    for (unsigned int i = 0; i < n; ++i) {
        state = state * 1103515245u + 12345u;
        v[i] = (state >> 8) & 0xFFFFu;
    }
    return v;
}

// CPU reference: sequential, order-preserving filter -- this IS the
// expected stable output, element-for-element.
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
    std::vector<unsigned int> ref = filterCPU(input_h);

    std::vector<unsigned int> gpu;
    unsigned int outputSize = 0;
    float ms = runFilter(input_h, gpu, outputSize);

    // Stable filter: verify EXACT order-preserving match, not just the set.
    bool ok = (outputSize == ref.size()) && (gpu == ref);
    printf("N=%u (blocks=%u): cpu kept=%zu gpu kept=%u  %.4f ms  [%s]\n",
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
