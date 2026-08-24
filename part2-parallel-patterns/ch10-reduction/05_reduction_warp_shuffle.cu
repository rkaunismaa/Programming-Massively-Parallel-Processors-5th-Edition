// Chapter 10: Reduction
// §10.7  Reducing synchronization overhead with warp-level primitives
//        (Fig. 10.12, Fig. 10.13, Fig. 10.14)
//
// In file 04's shared-memory kernel, once the active-thread count drops to
// 32 (i.e. down to a single warp), every remaining loop iteration still
// pays for a __syncthreads() barrier and shared-memory reads/writes even
// though all the action is now confined to one warp. §10.7's fix uses warp
// shuffle intrinsics -- direct register-to-register data exchange between
// threads of the SAME warp, no shared memory or __syncthreads() needed --
// for the final stage of the tree, once only one warp remains.
//
// Two small helper device functions (as the book defines them verbatim):
//   warpIdx() = threadIdx.x / WARP_SIZE   -- which warp a thread is in
//   laneIdx() = threadIdx.x % WARP_SIZE   -- a thread's index within its warp
//
// warp_reduce (Fig. 10.13) performs a complete 32-way sum reduction of a
// per-thread register value using __shfl_down_sync in a loop from stride 16
// down to 1 (5 = log2(32) steps): each thread adds in the value held by the
// thread `stride` lanes above it. After the loop, lane 0 of the warp holds
// the true sum of all 32 threads' input values (the other lanes' returned
// values are partial and not meaningful).
//
// The kernel (Fig. 10.14) is file 04's shared-memory kernel, but the loop
// stops once stride reaches the warp size 32 rather than continuing to 1
// ("the loop ... ends when the stride reaches the warp size, which is 32" --
// i.e. the stride==32 shared-memory step still runs, leaving exactly 32
// live partial sums in input_s[0..31]; the previously-remaining stride==16
// down to stride==1 steps are dropped from the shared-memory loop). After
// that, only warp 0 (warpIdx()==0) continues: each of its 32 threads loads
// its own input_s[threadIdx.x] into a register and the warp jointly calls
// warp_reduce, replacing what would otherwise have been 5 more
// __syncthreads()+shared-memory iterations with 5 shuffle-only steps.
// Thread 0 of the block writes the final sum to *output.
//
// This kernel is still single-block / N <= 2048, same as files 01, 02, 04.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

#define WARP_SIZE 32

// ---------------------------------------------------------------------------
// §10.7: warp/lane index helpers, defined verbatim as the book does.
// ---------------------------------------------------------------------------
__device__ unsigned int warpIdx() { return threadIdx.x / WARP_SIZE; }
__device__ unsigned int laneIdx() { return threadIdx.x % WARP_SIZE; }

// ---------------------------------------------------------------------------
// §10.7, Fig. 10.13: warp-wide sum reduction via __shfl_down_sync. Result is
// only meaningful in lane 0 of the calling warp.
// ---------------------------------------------------------------------------
__device__ float warp_reduce(float val) {
    float partialSum = val;
    for (unsigned int stride = WARP_SIZE / 2; stride >= 1; stride /= 2) {
        partialSum += __shfl_down_sync(0xffffffff, partialSum, stride);
    }
    return partialSum;
}

// ---------------------------------------------------------------------------
// §10.7, Fig. 10.14: shared-memory reduction down to one warp, then
// warp-shuffle reduction for the final 5 steps.
// ---------------------------------------------------------------------------
__global__ void reduction_warp_shuffle_kernel(const float *input, float *output) {
    extern __shared__ float input_s[];
    unsigned int t = threadIdx.x;

    input_s[t] = input[t] + input[t + blockDim.x];

    for (unsigned int stride = blockDim.x / 2; stride >= WARP_SIZE; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            input_s[t] += input_s[t + stride];
        }
    }

    if (warpIdx() == 0) {
        float partialSum = input_s[t];
        partialSum = warp_reduce(partialSum);
        if (t == 0) {
            *output = partialSum;
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

// Runs the warp-shuffle kernel once on an N-element input. N must be a
// power of two, 128 <= N <= 2048, so that blockDim.x = N/2 is a multiple of
// the warp size and >= 64 (the shared-memory loop's stride>=WARP_SIZE
// bound needs blockDim.x/2 >= WARP_SIZE to have anything to do; when
// blockDim.x == WARP_SIZE*2 == 64, the loop still runs its one stride==32
// iteration correctly). `input` is read-only, so one upload serves both
// the warm-up and timed launches.
float runWarpShuffleReduction(const std::vector<float> &input_h, float *sum_out) {
    unsigned int n = static_cast<unsigned int>(input_h.size());
    size_t bytes = n * sizeof(float);
    unsigned int blockDim = n / 2;
    size_t shmemBytes = blockDim * sizeof(float);

    float *input_d, *output_d;
    CUDA_CHECK(cudaMalloc((void **)&input_d, bytes));
    CUDA_CHECK(cudaMalloc((void **)&output_d, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(input_d, input_h.data(), bytes, cudaMemcpyHostToDevice));

    dim3 dimBlock(blockDim);
    dim3 dimGrid(1);

    reduction_warp_shuffle_kernel<<<dimGrid, dimBlock, shmemBytes>>>(input_d, output_d);  // warm-up
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    reduction_warp_shuffle_kernel<<<dimGrid, dimBlock, shmemBytes>>>(input_d, output_d);
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
    float ms = runWarpShuffleReduction(input_h, &gpuSum);

    bool ok = nearlyEqual(gpuSum, static_cast<float>(ref), 1e-2f);
    printf("N=%u: cpu=%.6f gpu=%.6f  %.4f ms  [%s]\n", n, ref, gpuSum, ms, ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    bool ok = true;
    // N a power of two with blockDim.x = N/2 >= WARP_SIZE (64), <= 2048.
    ok = runTestCase(128) && ok;
    ok = runTestCase(512) && ok;
    ok = runTestCase(2048) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
