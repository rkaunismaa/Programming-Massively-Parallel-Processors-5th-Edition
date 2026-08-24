// Chapter 10: Reduction
// §10.3  A simple reduction kernel (Fig. 10.5)
//
// A reduction derives a single value (here, a sum) from a list of values
// (§10.1). §10.2 shows that a sequential sum reduction takes O(N) work and
// O(N) steps (span), while a parallel reduction *tree* -- pairing up
// elements, then pairing up the partial sums, and so on -- performs the
// same O(N) work but only O(log N) steps, because floating-point addition
// is (for practical, tolerance-accepting purposes -- §10.2's own caveat
// about non-strict associativity of float addition) associative, so
// parentheses can be inserted at different positions of the same operand
// list without changing the (tolerably) same result.
//
// This file implements the book's *first, simplest* sum-reduction kernel
// (Fig. 10.5), which realizes the sum-reduction tree of Fig. 10.3(b) within
// a single thread block:
//   - For an N-element input (N a power of two, N <= 2048), the kernel is
//     launched with ONE block of N/2 threads (a block can have up to 1024
//     threads, so this kernel tops out at 2048 elements -- "We will
//     eliminate this limitation in Section 10.10").
//   - "Owner computes": thread t owns location input[2*t] (line 02) and is
//     the only thread that ever writes to it.
//   - stride starts at 1 and doubles each iteration (line 03) until it
//     exceeds blockDim.x. In iteration n, stride == 2^n.
//   - Only threads whose threadIdx.x is a multiple of stride are active
//     (line 04: `threadIdx.x % stride == 0`) and add the element `stride`
//     away into their owned location (line 05): input[i] += input[i +
//     stride].
//   - __syncthreads() (line 07) at the end of every iteration is both a
//     barrier AND (per §10.3's closing discussion) the memory fence that
//     makes the previous iteration's writes visible to the threads reading
//     them in the next iteration.
//   - After the last iteration only thread 0 is active, and input[0] holds
//     the total sum, which thread 0 writes to *output (line 10).
//
// §10.4 will show that this kernel suffers heavy control divergence (the
// active-thread test `threadIdx.x % stride == 0` scatters active threads
// across warps), and §10.5 will show it also suffers memory-access
// divergence (owned locations 2*t are never adjacent within a warp, so
// global-memory accesses are never coalesced). This file is deliberately
// the *un-optimized* baseline that files 02-08 progressively fix.
//
// IMPORTANT: like the book's kernel, this one overwrites the `input` array
// in global memory as it runs (the "owner computes" locations are
// themselves elements of the input array) -- the array passed in does NOT
// retain its original contents after the kernel returns.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

// ---------------------------------------------------------------------------
// §10.3, Fig. 10.5: the simple single-block sum-reduction kernel.
// ---------------------------------------------------------------------------
__global__ void reduction_naive_kernel(float *input, float *output) {
    unsigned int i = 2 * threadIdx.x;
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (threadIdx.x % stride == 0) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *output = input[0];
    }
}

// CPU reference, Fig. 10.1: sequential sum reduction, sum initialized to
// the addition identity 0.0 and accumulated left-to-right. Accumulated in
// double so the reference itself doesn't add extra rounding noise beyond
// what float32 tree-vs-sequential summation order already introduces (see
// runTestCase for how that's reconciled against the GPU's float32 result).
double reduction_cpu(const float *input, unsigned int n) {
    double sum = 0.0;
    for (unsigned int i = 0; i < n; ++i) {
        sum += input[i];
    }
    return sum;
}

// Deterministic synthetic input: uniform floats in [0, 1). Keeping values
// small and non-negative keeps the CPU-sequential vs. GPU-tree-order
// summation difference (see §10.2: float addition is not strictly
// associative, so reordering the parentheses can change the last few bits
// of the result) well within the tolerance used by nearlyEqual.
std::vector<float> generateInput(unsigned int n) {
    std::vector<float> v(n);
    unsigned int state = 987654321u;
    for (unsigned int i = 0; i < n; ++i) {
        state = state * 1103515245u + 12345u;
        v[i] = static_cast<float>((state >> 8) & 0xFFFFFF) / static_cast<float>(0x1000000);
    }
    return v;
}

// Runs the naive kernel once on an N-element input (N must be a power of
// two, N <= 2048 -- the single-block limit of Fig. 10.5). Because the
// kernel overwrites `input` in place, we re-upload a pristine copy before
// the untimed warm-up launch and again before the timed launch, so JIT
// warm-up cost is excluded from the timed measurement and both launches
// see the same original data.
float runNaiveReduction(const std::vector<float> &input_h, float *sum_out) {
    unsigned int n = static_cast<unsigned int>(input_h.size());
    size_t bytes = n * sizeof(float);

    float *input_d, *output_d;
    CUDA_CHECK(cudaMalloc((void **)&input_d, bytes));
    CUDA_CHECK(cudaMalloc((void **)&output_d, sizeof(float)));

    dim3 dimBlock(n / 2);
    dim3 dimGrid(1);

    CUDA_CHECK(cudaMemcpy(input_d, input_h.data(), bytes, cudaMemcpyHostToDevice));
    reduction_naive_kernel<<<dimGrid, dimBlock>>>(input_d, output_d);  // warm-up
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(input_d, input_h.data(), bytes, cudaMemcpyHostToDevice));
    GpuTimer timer;
    timer.start();
    reduction_naive_kernel<<<dimGrid, dimBlock>>>(input_d, output_d);
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
    float ms = runNaiveReduction(input_h, &gpuSum);

    bool ok = nearlyEqual(gpuSum, static_cast<float>(ref), 1e-2f);
    printf("N=%u: cpu=%.6f gpu=%.6f  %.4f ms  [%s]\n", n, ref, gpuSum, ms, ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    bool ok = true;
    // N must be a power of two <= 2048 (single block of N/2 <= 1024
    // threads), per Fig. 10.5's design.
    ok = runTestCase(128) && ok;
    ok = runTestCase(512) && ok;
    ok = runTestCase(2048) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
