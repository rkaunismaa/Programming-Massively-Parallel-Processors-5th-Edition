// Chapter 10: Reduction
// §10.10  Thread coarsening to reduce overhead (Fig. 10.19, Fig. 10.20)
//
// File 07's multi-block kernel launches N/2 threads for an N-element
// reduction -- maximum parallelism, but on hardware with limited execution
// resources, more thread blocks get launched than can run concurrently, and
// the excess blocks are simply serialized by the hardware scheduler. §10.10:
// "if the hardware were to serialize these thread blocks, we are better off
// serializing them ourselves in a more efficient manner" -- i.e. apply
// thread coarsening (as introduced generally in Ch. 6): give each thread
// block more elements to process serially per thread, so fewer blocks (and
// therefore fewer redundant per-block reduction-tree overhead episodes) are
// needed to cover the same input.
//
// Fig. 10.20's kernel is file 07's atomic multi-block kernel (Fig. 10.18)
// with exactly the two changes §10.10 describes:
//   - segment = blockIdx.x * COARSE_FACTOR * 2 * blockDim.x (line 03): the
//     block's segment is now COARSE_FACTOR times larger than file 07's
//     2*blockDim.x, since each thread will handle COARSE_FACTOR*2 elements
//     instead of 2.
//   - Rather than a single input[i] + input[i+blockDim.x] add, a coarsening
//     loop (lines 06-08) accumulates COARSE_FACTOR*2 elements, each
//     blockDim.x apart, into one register-only partialSum, with NO
//     __syncthreads() inside the loop ("no calls to __syncthreads() are
//     made in the loop because the threads act independently" -- every
//     thread just accumulates its own disjoint set of global-memory
//     elements). This is illustrated for COARSE_FACTOR=2 (4 elements/thread)
//     in Fig. 10.19; this file uses COARSE_FACTOR=4 (8 elements/thread),
//     matching this repo's Ch. 9 coarsening samples' factor.
// Everything after the coarsening loop (warp_reduce, the one shared-memory
// hop between stages, the second warp_reduce, and the atomicAdd into
// *output) is byte-for-byte identical to file 07 -- coarsening only changes
// how a thread arrives at its initial partialSum.
//
// §10.10's own qualitative argument (Fig. 10.21): two uncoarsened blocks
// serialized by the hardware collectively take 8 steps (2 fully-utilized +
// 6 divergent/synchronized), while one block coarsened by a factor of 2
// doing the same work takes only 6 steps (3 fully-utilized, uncoordinated
// coarsening-loop steps + 3 divergent/synchronized tree steps) -- fewer
// total steps AND a higher fraction of them are full-utilization work. This
// file demonstrates the concrete, book-derivable consequence of that
// argument for its own COARSE_FACTOR=4: covering the same N elements as
// file 07 takes exactly 1/COARSE_FACTOR as many blocks (and therefore 1/4
// as many atomicAdd contentions on *output and 1/4 as many per-block
// reduction-tree/shared-memory episodes) -- see the README for the
// measured block counts.
//
// Same N-must-be-an-exact-multiple-of-segment-size assumption as file 07
// (handling a remainder is Exercise 5, out of this project's scope).

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

#define WARP_SIZE 32
#define BLOCK_DIM 1024
#define COARSE_FACTOR 4

__device__ unsigned int warpIdx() { return threadIdx.x / WARP_SIZE; }
__device__ unsigned int laneIdx() { return threadIdx.x % WARP_SIZE; }

// §10.7, Fig. 10.13 (duplicated per this repo's self-contained-file
// convention).
__device__ float warp_reduce(float val) {
    float partialSum = val;
    for (unsigned int stride = WARP_SIZE / 2; stride >= 1; stride /= 2) {
        partialSum += __shfl_down_sync(0xffffffff, partialSum, stride);
    }
    return partialSum;
}

// ---------------------------------------------------------------------------
// §10.10, Fig. 10.20: coarsened multi-block sum-reduction kernel, built
// directly on file 07's atomic two-stage warp-wide kernel.
// ---------------------------------------------------------------------------
__global__ void reduction_coarsened_kernel(const float *input, float *output) {
    __shared__ float shared[WARP_SIZE];
    unsigned int t = threadIdx.x;
    unsigned int segment = blockIdx.x * COARSE_FACTOR * 2 * blockDim.x;
    unsigned int i = segment + t;

    if (t < WARP_SIZE) {
        shared[t] = 0.0f;
    }
    __syncthreads();

    float partialSum = input[i];
    for (unsigned int c = 1; c < COARSE_FACTOR * 2; ++c) {
        partialSum += input[i + c * blockDim.x];
    }
    partialSum = warp_reduce(partialSum);

    if (laneIdx() == 0) {
        shared[warpIdx()] = partialSum;
    }
    __syncthreads();

    if (warpIdx() == 0) {
        float val = shared[t];
        val = warp_reduce(val);
        if (t == 0) {
            atomicAdd(output, val);
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

// Runs the coarsened kernel on an N-element input. N must be an exact
// multiple of the per-block segment size COARSE_FACTOR*2*BLOCK_DIM = 8192.
float runCoarsenedReduction(const std::vector<float> &input_h, float *sum_out) {
    unsigned int n = static_cast<unsigned int>(input_h.size());
    size_t bytes = n * sizeof(float);
    unsigned int segmentSize = COARSE_FACTOR * 2 * BLOCK_DIM;
    unsigned int numBlocks = n / segmentSize;

    float *input_d, *output_d;
    CUDA_CHECK(cudaMalloc((void **)&input_d, bytes));
    CUDA_CHECK(cudaMalloc((void **)&output_d, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(input_d, input_h.data(), bytes, cudaMemcpyHostToDevice));

    dim3 dimBlock(BLOCK_DIM);
    dim3 dimGrid(numBlocks);

    CUDA_CHECK(cudaMemset(output_d, 0, sizeof(float)));
    reduction_coarsened_kernel<<<dimGrid, dimBlock>>>(input_d, output_d);  // warm-up
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemset(output_d, 0, sizeof(float)));
    GpuTimer timer;
    timer.start();
    reduction_coarsened_kernel<<<dimGrid, dimBlock>>>(input_d, output_d);
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
    float ms = runCoarsenedReduction(input_h, &gpuSum);

    unsigned int numBlocks = n / (COARSE_FACTOR * 2 * BLOCK_DIM);
    bool ok = nearlyEqual(gpuSum, static_cast<float>(ref), 1e-2f);
    printf("N=%u (%u blocks): cpu=%.6f gpu=%.6f  %.4f ms  [%s]\n", n, numBlocks, ref, gpuSum, ms,
           ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    bool ok = true;
    // N must be an exact multiple of COARSE_FACTOR*2*BLOCK_DIM = 8192. The
    // 1,024,000-element case matches file 07's largest test case exactly
    // (2048*500 == 8192*125), so the README's block-count comparison
    // (500 vs. 125 blocks, a COARSE_FACTOR=4 reduction) is apples to apples.
    ok = runTestCase(8192 * 10) && ok;   // 81,920 elements, 10 blocks
    ok = runTestCase(8192 * 125) && ok;  // 1,024,000 elements, 125 blocks

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
