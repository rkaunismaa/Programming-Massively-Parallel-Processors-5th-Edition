// Chapter 10: Reduction
// §10.5  Reducing memory access divergence
//
// §10.5 introduces no new kernel listing/figure of its own -- it instead
// revisits the SAME two kernels already shown (Fig. 10.5's naive kernel and
// Fig. 10.8's control-divergence-reduced kernel from file 02) and analyzes
// them from a different angle: global-memory coalescing. This file is a
// head-to-head comparison of exactly those two kernels' memory-access
// behavior, which is what §10.5 itself does in prose.
//
// The naive kernel (Fig. 10.5, i = 2*threadIdx.x) has adjacent threads
// owning locations that are 2 apart, and each iteration's "reach stride
// away" read lands even further apart as stride grows. Adjacent threads in
// a warp therefore never access adjacent memory, so accesses are never
// coalesced: "twice the number of global memory transactions as that for a
// coalesced access are triggered, and half the data returned will not be
// used" in the very first iteration alone, growing worse each iteration.
// §10.5 derives the exact total for a 256-element input:
//     (N/64*5*2 + N/64 + N/64/2 + N/64/4 + N/64/8 + 1) * 3
//         = (4*5*2 + 4 + 2 + 1) * 3 = 141 global memory requests
// (N/64 = 4 warps; the first 5 iterations each have >=2 active threads per
// warp so each active warp issues 2 divergent requests; the final 5
// iterations have exactly 1 active thread per active warp, issuing 1
// request, with warps halving each iteration; the *3 accounts for 2 reads
// + 1 write per active thread per iteration).
//
// The convergent kernel (Fig. 10.8, i = threadIdx.x) has adjacent threads
// always owning adjacent locations, so "the adjacent threads in each warp
// always access adjacent locations in the global memory so the accesses
// are always coalesced" -- §10.5 does not give a second worked total (it
// only asserts coalescing qualitatively and that execution time "is likely
// to be significantly better"), so this file does NOT fabricate a
// competing request count for it; instead it reports the one number the
// book actually derives (141, for N=256) and lets the timing comparison
// speak to the real-hardware effect.
//
// This file times BOTH kernels in the same process, so per this project's
// timing-fairness rule, each gets its own untimed warm-up launch
// immediately before its own timed launch, and both do the exact same
// amount of work (same N, same number of adds) -- only the memory-access
// pattern differs.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

// ---------------------------------------------------------------------------
// Fig. 10.5: naive kernel, i = 2*threadIdx.x, uncoalesced accesses.
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

// ---------------------------------------------------------------------------
// Fig. 10.8: convergent kernel, i = threadIdx.x, coalesced accesses.
// ---------------------------------------------------------------------------
__global__ void reduction_convergent_kernel(float *input, float *output) {
    unsigned int i = threadIdx.x;
    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *output = input[0];
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

// §10.5's own worked total of global memory requests for the naive kernel
// (Fig. 10.5) on a 256-element input, reproduced verbatim from the book's
// arithmetic (not a general closed-form -- the book only derives this one
// case): (N/64*5*2 + N/64 + N/64/2 + N/64/4 + ... + 1) * 3, where the
// second parenthesized term is the geometric series N/64, N/64/2, ...,
// down to (and including) 1, i.e. the halving-warp-count tail the book
// describes.
static unsigned int bookNaiveMemoryRequests256() {
    unsigned int warps = 256u / 64u;  // 4
    unsigned int tail = 0u;
    for (unsigned int w = warps; w >= 1u; w /= 2u) {
        tail += w;
    }
    unsigned int sum = warps * 5u * 2u + tail;
    return sum * 3u;  // = (40 + 7) * 3 = 141
}

// Runs one kernel on `input_h`, with its own untimed warm-up launch
// immediately before its own timed launch (both re-upload pristine data
// first, since both kernels mutate `input` in place). Returns the timed
// duration in ms and writes the result sum to *sum_out.
template <typename KernelFn>
float runKernel(KernelFn kernel, const std::vector<float> &input_h, float *sum_out) {
    unsigned int n = static_cast<unsigned int>(input_h.size());
    size_t bytes = n * sizeof(float);

    float *input_d, *output_d;
    CUDA_CHECK(cudaMalloc((void **)&input_d, bytes));
    CUDA_CHECK(cudaMalloc((void **)&output_d, sizeof(float)));

    dim3 dimBlock(n / 2);
    dim3 dimGrid(1);

    CUDA_CHECK(cudaMemcpy(input_d, input_h.data(), bytes, cudaMemcpyHostToDevice));
    kernel<<<dimGrid, dimBlock>>>(input_d, output_d);  // warm-up
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(input_d, input_h.data(), bytes, cudaMemcpyHostToDevice));
    GpuTimer timer;
    timer.start();
    kernel<<<dimGrid, dimBlock>>>(input_d, output_d);
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

    float naiveSum = 0.0f, convSum = 0.0f;
    float naiveMs = runKernel(reduction_naive_kernel, input_h, &naiveSum);
    float convMs = runKernel(reduction_convergent_kernel, input_h, &convSum);

    bool okNaive = nearlyEqual(naiveSum, static_cast<float>(ref), 1e-2f);
    bool okConv = nearlyEqual(convSum, static_cast<float>(ref), 1e-2f);

    printf("N=%u: cpu=%.6f  naive=%.6f (%.4f ms) [%s]  convergent=%.6f (%.4f ms) [%s]\n", n, ref, naiveSum, naiveMs,
           okNaive ? "match" : "MISMATCH", convSum, convMs, okConv ? "match" : "MISMATCH");

    if (n == 256) {
        printf("  Book's §10.5 worked example (Fig. 10.5, N=256): %u global memory requests"
               " (uncoalesced) -- reproduced from the text's own arithmetic, not measured here.\n",
               bookNaiveMemoryRequests256());
    }

    return okNaive && okConv;
}

int main() {
    bool ok = true;
    // Same single-block power-of-two constraint as files 01/02. N=256 is
    // included specifically because it's the exact case §10.5 works out
    // numerically in the book text.
    ok = runTestCase(256) && ok;
    ok = runTestCase(512) && ok;
    ok = runTestCase(2048) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
