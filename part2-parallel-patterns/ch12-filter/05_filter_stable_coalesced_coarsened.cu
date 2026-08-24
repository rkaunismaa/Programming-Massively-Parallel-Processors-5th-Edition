// Chapter 12: Filter
// §12.6  Improving memory coalescing with shared memory and thread
//        coarsening (Figs. 12.9 - 12.10)
//
// Book-fidelity disclosure: §12.6 gives no full kernel code -- it presents
// the optimization purely through two worked diagrams and closes with "We
// leave the detailed stable filter implementation with exclusive scan,
// privatization, and thread coarsening as an exercise." What IS in the main
// text, and what this file implements, are the diagrams' concrete mechanics
// (matching this project's Chapter 11 precedent: a described-but-uncoded
// mechanism is in scope, an undescribed exercise prompt is not):
//
//   Fig. 12.9 (coalescing via shared memory, no coarsening yet): file 04's
//   kernel writes each surviving key directly to its final global address,
//   so a block's kept keys land at scattered offsets separated by however
//   many keys OTHER threads/warps contributed in between -- e.g. globally
//   adjacent output slots 2 and 3 (k4, k6) are written by different warps
//   with inactive lanes between them. Fig. 12.9's fix: gather a block's kept
//   keys into a per-block PRIVATE compacted list in shared memory first
//   (exactly file 03's privatization idea, but applied to the stable
//   kernel), then have the block's threads write that shared buffer to
//   global memory as ONE contiguous run, consecutive threads to consecutive
//   addresses.
//
//   Fig. 12.10 (thread coarsening on top of that): give each block more
//   input keys than it has threads (COARSE_FACTOR keys/thread, contiguous
//   per thread, matching the worked N=16/2-blocks-of-4-threads example --
//   block 0 covers k0..k7, thread 0 covers k0-k1, thread 1 covers k2-k3,
//   etc.) so each block's compacted shared-memory run, and hence each
//   contiguous global write, is larger and there are fewer of them. §12.6
//   also notes coarsening's larger, and arguably more important, benefit:
//   it coarsens the grid-wide SCAN itself (Ch. 11 §11.6/11.7), which file
//   04's single-lookback scan inherits for free once each thread folds
//   COARSE_FACTOR keep-bits into one local count before entering the scan.
//
// Built directly on file 04's grid-wide exclusive-scan machinery
// (warpScan/blockScan/interBlockScan, unchanged) with each thread now
// handling COARSE_FACTOR contiguous input keys instead of one.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda/atomic>

#include "../../common/cuda_utils.h"

#define WARP_SIZE 32
#define COARSE_FACTOR 4
#define BLOCK_DIM 256

__device__ __host__ bool cond(unsigned int val) { return (val % 2u) == 0u; }

__device__ unsigned int warpIdx() { return threadIdx.x / WARP_SIZE; }
__device__ unsigned int laneIdx() { return threadIdx.x % WARP_SIZE; }

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

// Same single-lookback inter-block scan as file 04 (Ch. 11 §11.9,
// Fig. 11.17); only the leader thread's `val` is ever read.
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
// §12.6: stable filter with shared-memory coalescing (Fig. 12.9) and thread
// coarsening (Fig. 12.10), built on file 04's grid-wide scan.
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
    __shared__ unsigned int output_s[COARSE_FACTOR * BLOCK_DIM];

    unsigned int segment = bid * blockDim.x * COARSE_FACTOR;
    unsigned int threadSegment = threadIdx.x * COARSE_FACTOR;

    // Each thread sequentially filters its own COARSE_FACTOR-element
    // contiguous chunk (Fig. 12.10: thread 0 -> k0,k1; thread 1 -> k2,k3;
    // ...), buffering the surviving values in registers and counting them.
    unsigned int threadVals[COARSE_FACTOR];
    unsigned int threadKeepCount = 0;
#pragma unroll
    for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        unsigned int idx = segment + threadSegment + c;
        if (idx < N) {
            unsigned int val = input[idx];
            if (cond(val)) {
                threadVals[threadKeepCount] = val;
                ++threadKeepCount;
            }
        }
    }

    // Block-local exclusive scan of each thread's private keep-count gives
    // every thread its base offset into the block's compacted shared-memory
    // list (Fig. 12.9's "private output" row).
    unsigned int inclusiveLocal = blockScan(threadKeepCount, warpSums_s);
    unsigned int threadBase = inclusiveLocal - threadKeepCount;

#pragma unroll
    for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        if (c < threadKeepCount) {
            output_s[threadBase + c] = threadVals[c];
        }
    }
    __syncthreads();

    // Reserve this block's contiguous run in the PUBLIC output array via
    // grid-wide single lookback (leader thread's inclusiveLocal == this
    // block's total keep count).
    unsigned int previousBlockSum = interBlockScan(inclusiveLocal, bid, partialSums, flags);

    __shared__ unsigned int blockTotal_s;
    if (threadIdx.x == blockDim.x - 1) {
        blockTotal_s = inclusiveLocal;
    }
    __syncthreads();

    // Coalesced write: consecutive threads write consecutive global
    // addresses, looping if the block's compacted run is larger than the
    // block itself (Fig. 12.9/12.10's global-memory row, written as one
    // contiguous chunk per block).
    for (unsigned int base = 0; base < blockTotal_s; base += blockDim.x) {
        unsigned int idx = base + threadIdx.x;
        if (idx < blockTotal_s) {
            output[previousBlockSum + idx] = output_s[idx];
        }
    }

    // The block owning the last logical segment (highest bid) knows the
    // final output size once its own scan is done.
    if (bid == gridDim.x - 1 && threadIdx.x == blockDim.x - 1) {
        *outputSize = previousBlockSum + blockTotal_s;
    }
}

std::vector<unsigned int> generateInput(unsigned int n) {
    std::vector<unsigned int> v(n);
    unsigned int state = 523456789u;
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
    unsigned int segSize = COARSE_FACTOR * BLOCK_DIM;
    unsigned int numBlocks = (n + segSize - 1) / segSize;

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

    unsigned int segSize = COARSE_FACTOR * BLOCK_DIM;
    unsigned int numBlocks = (n + segSize - 1) / segSize;

    bool ok = (outputSize == ref.size()) && (gpu == ref);
    printf("N=%u (blocks=%u, COARSE_FACTOR=%d): cpu kept=%zu gpu kept=%u  %.4f ms  [%s]\n",
           n, numBlocks, COARSE_FACTOR, ref.size(), outputSize, ms, ok ? "match" : "MISMATCH");
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
