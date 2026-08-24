// Chapter 10: Reduction
// §10.9  Reduction for arbitrary length inputs (Fig. 10.17, Fig. 10.18)
//
// Every kernel so far (files 01-06) assumes a SINGLE thread block, because
// __syncthreads() only synchronizes threads within one block, capping input
// size at 2*1024 = 2048 elements. §10.9 removes that limit by partitioning
// the input into per-block segments that each execute the file 06 two-stage
// warp-wide reduction independently, then combining every block's partial
// sum into the final result with an atomic add (Fig. 10.17): "All blocks
// then independently execute a reduction tree and accumulate their results
// to the final output using an atomic add operation."
//
// Fig. 10.18's kernel has "only a few changes compared to the kernel in
// Fig. 10.16" (this chapter's file 06):
//   - segment = blockIdx.x * 2 * blockDim.x (line 03): each block owns a
//     2*blockDim.x-element slice of the input (blockDim.x threads, 2
//     elements per thread, exactly as in file 06). For 1024-thread blocks,
//     segment size is 2048 -- segment starts are 0, 2048, 4096, ... for
//     blocks 0, 1, 2, ...
//   - i = segment + threadIdx.x (line 04) replaces file 06's plain t as the
//     owned global-array index; the rest of the per-block two-stage
//     warp-wide reduction is unchanged (§10.9: "One can adapt any of the
//     single-block reduction kernel code into this part of the kernel").
//   - The final write becomes an ATOMIC add into the (shared, cross-block)
//     *output rather than a plain store (lines 18-20): "instead of having
//     Thread 0 ... writing the partial sum to the global output value, it
//     accumulates the partial sum to the output using an atomic addition
//     operation." §10.9 stresses this requires the reduction operator to be
//     both commutative and associative, since "the thread blocks can make
//     their contributions in any arbitrary order."
//
// The book's Fig. 10.18 assumes N is an exact multiple of the per-block
// segment size (handling a remainder is left to Exercise 5, which per this
// project's scope is a reader exercise, not chapter content) -- this file
// follows that same assumption and only tests N values that are exact
// multiples of 2*blockDim.x.
//
// The host must zero *output before launch (the atomic add accumulates
// onto whatever is already there).

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

#define WARP_SIZE 32
#define BLOCK_DIM 1024

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
// §10.9, Fig. 10.18: multi-block sum reduction using atomic accumulation,
// built directly on file 06's two-stage warp-wide per-block reduction.
// ---------------------------------------------------------------------------
__global__ void reduction_atomic_kernel(const float *input, float *output) {
    __shared__ float shared[WARP_SIZE];
    unsigned int t = threadIdx.x;
    unsigned int segment = blockIdx.x * 2 * blockDim.x;
    unsigned int i = segment + t;

    if (t < WARP_SIZE) {
        shared[t] = 0.0f;
    }
    __syncthreads();

    float partialSum = input[i] + input[i + blockDim.x];
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

// Runs the multi-block atomic-accumulation kernel on an N-element input. N
// must be an exact multiple of the per-block segment size 2*BLOCK_DIM =
// 2048, matching Fig. 10.18's own assumption (see header comment).
float runAtomicReduction(const std::vector<float> &input_h, float *sum_out) {
    unsigned int n = static_cast<unsigned int>(input_h.size());
    size_t bytes = n * sizeof(float);
    unsigned int segmentSize = 2 * BLOCK_DIM;
    unsigned int numBlocks = n / segmentSize;

    float *input_d, *output_d;
    CUDA_CHECK(cudaMalloc((void **)&input_d, bytes));
    CUDA_CHECK(cudaMalloc((void **)&output_d, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(input_d, input_h.data(), bytes, cudaMemcpyHostToDevice));

    dim3 dimBlock(BLOCK_DIM);
    dim3 dimGrid(numBlocks);

    CUDA_CHECK(cudaMemset(output_d, 0, sizeof(float)));
    reduction_atomic_kernel<<<dimGrid, dimBlock>>>(input_d, output_d);  // warm-up
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemset(output_d, 0, sizeof(float)));
    GpuTimer timer;
    timer.start();
    reduction_atomic_kernel<<<dimGrid, dimBlock>>>(input_d, output_d);
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
    float ms = runAtomicReduction(input_h, &gpuSum);

    unsigned int numBlocks = n / (2 * BLOCK_DIM);
    bool ok = nearlyEqual(gpuSum, static_cast<float>(ref), 1e-2f);
    printf("N=%u (%u blocks): cpu=%.6f gpu=%.6f  %.4f ms  [%s]\n", n, numBlocks, ref, gpuSum, ms,
           ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    bool ok = true;
    // N must be an exact multiple of 2*BLOCK_DIM = 2048 (Fig. 10.18's own
    // assumption). These sizes go well past the 2048-element single-block
    // ceiling of files 01-06, demonstrating the "arbitrary length" this
    // section's multi-block design enables.
    ok = runTestCase(2048 * 10) && ok;      // 20,480 elements, 10 blocks
    ok = runTestCase(2048 * 500) && ok;     // 1,024,000 elements, 500 blocks

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
