// Chapter 12: Filter
// §12.7  In-place stable filter (Fig. 12.11)
//
// So far every stable-filter kernel (files 04-05) has written into a
// SEPARATE output array. §12.7 asks: what if input and output must be the
// SAME array (the chapter's own motivating example: compacting a garbage
// collector's heap in place because there isn't room for a second copy)?
// The danger is a thread overwriting a value another thread hasn't read
// yet -- e.g. in Fig. 12.10's worked example, k8's output slot is the same
// memory cell k4 used to occupy, so whoever reads k4 must finish before
// anyone writes k8 there.
//
// §12.7's key claim is that the single-lookback stable-filter kernel from
// §12.5/§12.9 ALREADY enforces the ordering an in-place filter needs,
// without any code change:
//   - WITHIN a block: every thread loads its own input value (register)
//     before the kernel does any writing, and a barrier synchronization
//     between the read step and the write step (this file makes that
//     barrier explicit, immediately after the load) guarantees no thread's
//     write can race a same-block thread's still-pending read.
//   - ACROSS blocks: Fig. 12.11's dependence diagram shows read_i ->
//     scan_i -> scan_j -> write_j for any j > i. Block j's leader thread
//     cannot pass interBlockScan's lookback wait until block i's leader has
//     already published its partial sum -- which block i only does AFTER
//     block i's own read step. So block j can never write ahead of block
//     i's reads, for every earlier block i, exactly the ordering an
//     in-place compaction requires. This is the same single-lookback
//     machinery from files 04-05, completely unmodified.
//
// This file is therefore file 04's kernel verbatim, called with the output
// pointer equal to the input pointer, plus the explicit read/write barrier
// §12.7 calls out. Correctness is checked by filtering into the SAME
// device buffer and comparing the surviving prefix against a CPU stable
// filter reference.

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

// Same single-lookback inter-block scan as files 04-05 (Ch. 11 §11.9,
// Fig. 11.17). Per §12.7, this is exactly the mechanism that makes the
// cross-block in-place ordering (Fig. 12.11) safe: block j's leader cannot
// get past the `while` wait below until block i (i < j) has already
// published `partialSums[i]` -- which only happens after block i has
// finished reading its own input values.
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
// §12.7: in-place stable filter. `data` serves as BOTH input and output.
// ---------------------------------------------------------------------------
__global__ void filterKernelInPlace(unsigned int *data, unsigned int N, unsigned int *outputSize,
                                     unsigned int *partialSums, unsigned int *flags,
                                     unsigned int *blockCounter) {
    __shared__ unsigned int bid_s;
    if (threadIdx.x == 0) {
        bid_s = atomicAdd(blockCounter, 1u);
    }
    __syncthreads();
    unsigned int bid = bid_s;

    __shared__ unsigned int warpSums_s[BLOCK_DIM / WARP_SIZE];

    unsigned int i = bid * blockDim.x + threadIdx.x;
    unsigned int val = (i < N) ? data[i] : 0u;
    unsigned int keep = (i < N && cond(val)) ? 1u : 0u;

    // §12.7's read/write barrier: every thread in this block has now READ
    // its input value into a register. No thread may WRITE into this
    // block's segment until every thread reaches this point.
    __syncthreads();

    unsigned int inclusiveLocal = blockScan(keep, warpSums_s);
    unsigned int previousBlockSum = interBlockScan(inclusiveLocal, bid, partialSums, flags);
    unsigned int offset = previousBlockSum + (inclusiveLocal - keep);

    if (keep) {
        data[offset] = val;
    }
    if (i == N - 1) {
        *outputSize = offset + keep;
    }
}

std::vector<unsigned int> generateInput(unsigned int n) {
    std::vector<unsigned int> v(n);
    unsigned int state = 623456791u;
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

float runFilterInPlace(const std::vector<unsigned int> &input_h, std::vector<unsigned int> &output_h,
                        unsigned int &outputSize_h) {
    unsigned int n = static_cast<unsigned int>(input_h.size());
    size_t bytes = n * sizeof(unsigned int);
    unsigned int numBlocks = (n + BLOCK_DIM - 1) / BLOCK_DIM;

    unsigned int *data_d, *outputSize_d, *partialSums_d;
    unsigned int *flags_d, *blockCounter_d;
    CUDA_CHECK(cudaMalloc((void **)&data_d, bytes));
    CUDA_CHECK(cudaMalloc((void **)&outputSize_d, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc((void **)&partialSums_d, numBlocks * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc((void **)&flags_d, numBlocks * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc((void **)&blockCounter_d, sizeof(unsigned int)));

    dim3 dimBlock(BLOCK_DIM);
    dim3 dimGrid(numBlocks);

    auto resetState = [&]() {
        CUDA_CHECK(cudaMemcpy(data_d, input_h.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(flags_d, 0, numBlocks * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(blockCounter_d, 0, sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(outputSize_d, 0, sizeof(unsigned int)));
    };

    resetState();
    filterKernelInPlace<<<dimGrid, dimBlock>>>(data_d, n, outputSize_d, partialSums_d, flags_d,
                                               blockCounter_d);  // warm-up
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    resetState();  // re-upload the untouched input before the timed, in-place run
    GpuTimer timer;
    timer.start();
    filterKernelInPlace<<<dimGrid, dimBlock>>>(data_d, n, outputSize_d, partialSums_d, flags_d,
                                               blockCounter_d);
    CUDA_CHECK(cudaGetLastError());
    float ms = timer.stopAndGetMs();

    CUDA_CHECK(cudaMemcpy(&outputSize_h, outputSize_d, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    output_h.resize(outputSize_h);
    if (outputSize_h > 0) {
        CUDA_CHECK(cudaMemcpy(output_h.data(), data_d, outputSize_h * sizeof(unsigned int),
                               cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaFree(data_d));
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
    float ms = runFilterInPlace(input_h, gpu, outputSize);

    bool ok = (outputSize == ref.size()) && (gpu == ref);
    printf("N=%u (blocks=%u): cpu kept=%zu gpu kept=%u (in-place)  %.4f ms  [%s]\n",
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
