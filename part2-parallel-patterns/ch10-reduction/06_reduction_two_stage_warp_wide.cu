// Chapter 10: Reduction
// §10.8  Further reducing synchronization overhead with two-stage
//        warp-wide reduction (Fig. 10.15, Fig. 10.16)
//
// File 05 eliminates shared-memory/barrier overhead from the LAST few
// (log2(32) = 5) steps of the tree, once the computation has narrowed down
// to a single warp -- but the FIRST several steps (while multiple warps are
// still active) still pay full shared-memory + __syncthreads() cost every
// iteration. §10.8's idea: have EVERY warp independently warp-reduce its
// own segment of the input FIRST (stage 1, no shared memory or
// __syncthreads() at all -- pure register shuffles), then have a single
// warp combine the resulting per-warp partial sums (stage 2).
//
// Fig. 10.16's kernel:
//   - Each thread loads and adds its two original elements directly from
//     global memory into a register (line 04) -- exactly like the first,
//     outside-the-loop step of files 04/05, but the result stays in a
//     register instead of being written to shared memory.
//   - ALL warps immediately call warp_reduce (Fig. 10.13) on that register
//     value (line 05) -- every warp in the block does this independently
//     and in parallel; unlike file 05's shared-memory phase, no warp ever
//     sits out control-divergence-free while other warps still work, but
//     also no warp ever touches shared memory or a barrier during this
//     phase either.
//   - The thread at lane 0 of each warp now holds that warp's total
//     partial sum, and writes it into shared memory at a slot indexed by
//     the warp's index (lines 07-10): shared[warpIdx()] = partialSum.
//   - A single __syncthreads() (line 11) is the ONLY block-wide barrier in
//     the whole kernel, needed once to make all warps' shared-memory writes
//     visible before stage 2 reads them.
//   - Only warp 0 continues (line 13): each of its 32 threads loads one
//     warp's partial sum from shared[threadIdx.x] (line 14) and the warp
//     jointly calls warp_reduce again (line 15) to combine up to 32
//     per-warp partial sums into the final total, which thread 0 writes to
//     *output (lines 16-18).
//
// §10.8's own tradeoff analysis: this design eliminates shared-memory
// access and __syncthreads() from stage 1 entirely (down to the single
// barrier between stages), at the cost of MORE control divergence during
// stage 1 (every warp's warp_reduce loop still has all 32 lanes technically
// "active" performing shuffles even once fewer than 32 lanes hold live
// data -- there's no early-exit the way file 05's shared-memory loop lets
// whole warps drop out). The book argues this trade is worthwhile because
// the reduction's later stages are latency-bound rather than
// occupancy-bound, so removing long-latency __syncthreads()/shared-memory
// operations helps more than removing (already latency-hidden) divergent
// shuffles would.
//
// Still single-block / N <= 2048; §10.9 (file 07) removes that limit.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

#define WARP_SIZE 32

__device__ unsigned int warpIdx() { return threadIdx.x / WARP_SIZE; }
__device__ unsigned int laneIdx() { return threadIdx.x % WARP_SIZE; }

// §10.7, Fig. 10.13 (same device function as file 05, duplicated per this
// repo's self-contained-file convention).
__device__ float warp_reduce(float val) {
    float partialSum = val;
    for (unsigned int stride = WARP_SIZE / 2; stride >= 1; stride /= 2) {
        partialSum += __shfl_down_sync(0xffffffff, partialSum, stride);
    }
    return partialSum;
}

// ---------------------------------------------------------------------------
// §10.8, Fig. 10.16: two-stage warp-wide sum-reduction kernel.
//
// `shared` has WARP_SIZE (32) slots, enough for up to 32 warps (the
// maximum for a 1024-thread block). When blockDim.x has fewer than 32
// warps, the unused high slots are explicitly zeroed first so warp 0's
// stage-2 warp_reduce (which unconditionally reads shared[0..31]) sums in
// harmless zeros for the slots no real warp wrote -- a defensive addition
// on top of the book's kernel, needed only because this file also
// exercises block sizes smaller than the book's full 1024-thread example.
// ---------------------------------------------------------------------------
__global__ void reduction_two_stage_kernel(const float *input, float *output) {
    __shared__ float shared[WARP_SIZE];
    unsigned int t = threadIdx.x;

    if (t < WARP_SIZE) {
        shared[t] = 0.0f;
    }
    __syncthreads();

    float partialSum = input[t] + input[t + blockDim.x];
    partialSum = warp_reduce(partialSum);

    if (laneIdx() == 0) {
        shared[warpIdx()] = partialSum;
    }
    __syncthreads();

    if (warpIdx() == 0) {
        float val = shared[t];
        val = warp_reduce(val);
        if (t == 0) {
            *output = val;
        }
    }
}

// CPU reference, Fig. 10.1 (see file 01).
double reduction_cpu(const float *input, unsigned int n) {
    double sum = 0.0;
    for (unsigned int i = 0; i < n; ++i) {
        sum += input[i];
    }
    return sum;
}

std::vector<float> generateInput(unsigned int n) {
    std::vector<float> v(n);
    unsigned int state = 987654321u;
    for (unsigned int i = 0; i < n; ++i) {
        state = state * 1103515245u + 12345u;
        v[i] = static_cast<float>((state >> 8) & 0xFFFFFF) / static_cast<float>(0x1000000);
    }
    return v;
}

// Runs the two-stage kernel once on an N-element input (N a power of two,
// blockDim.x = N/2 a multiple of WARP_SIZE, blockDim.x <= 1024 => N <= 2048).
float runTwoStageReduction(const std::vector<float> &input_h, float *sum_out) {
    unsigned int n = static_cast<unsigned int>(input_h.size());
    size_t bytes = n * sizeof(float);
    unsigned int blockDim = n / 2;

    float *input_d, *output_d;
    CUDA_CHECK(cudaMalloc((void **)&input_d, bytes));
    CUDA_CHECK(cudaMalloc((void **)&output_d, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(input_d, input_h.data(), bytes, cudaMemcpyHostToDevice));

    dim3 dimBlock(blockDim);
    dim3 dimGrid(1);

    reduction_two_stage_kernel<<<dimGrid, dimBlock>>>(input_d, output_d);  // warm-up
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    reduction_two_stage_kernel<<<dimGrid, dimBlock>>>(input_d, output_d);
    CUDA_CHECK(cudaGetLastError());
    float ms = timer.stopAndGetMs();

    CUDA_CHECK(cudaMemcpy(sum_out, output_d, sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(input_d));
    CUDA_CHECK(cudaFree(output_d));
    return ms;
}

bool runTestCase(unsigned int n) {
    std::vector<float> input_h = generateInput(n);
    double ref = reduction_cpu(input_h.data(), n);

    float gpuSum = 0.0f;
    float ms = runTwoStageReduction(input_h, &gpuSum);

    bool ok = nearlyEqual(gpuSum, static_cast<float>(ref), 1e-2f);
    printf("N=%u: cpu=%.6f gpu=%.6f  %.4f ms  [%s]\n", n, ref, gpuSum, ms, ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    bool ok = true;
    // Exercise block sizes from 1 warp/segment up to the book's full
    // 32-warp/1024-thread example.
    ok = runTestCase(256) && ok;   // blockDim.x=128 (4 warps)
    ok = runTestCase(512) && ok;   // blockDim.x=256 (8 warps)
    ok = runTestCase(2048) && ok;  // blockDim.x=1024 (32 warps)

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
