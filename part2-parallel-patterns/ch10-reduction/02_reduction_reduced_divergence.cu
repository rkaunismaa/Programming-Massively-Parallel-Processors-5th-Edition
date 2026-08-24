// Chapter 10: Reduction
// §10.4  Reducing control divergence (Fig. 10.8)
//
// The naive kernel of §10.3/Fig. 10.5 assigns thread t to owned position
// 2*t and tests `threadIdx.x % stride == 0` to pick active threads each
// iteration. §10.4 measures the cost of this: for a 256-element input, the
// naive kernel consumes (4*6 + 2 + 1)*32 = 864 units of execution resource
// to commit only 255 useful results, an efficiency of 255/864 = 0.30 (only
// 30% of consumed resources do useful work) -- because active threads
// become increasingly scattered across warps as the stride grows, so a
// whole warp stays "active" (and consumes full SIMD resources) even when
// only one of its 32 threads is doing real work.
//
// Fig. 10.8's fix keeps the SAME amount of computation and the SAME number
// of active threads per iteration, but changes WHICH threads are active
// and WHERE they write, so that active threads stay packed into a
// contiguous, shrinking prefix of the block rather than spreading out:
//   - Owner position is now i = threadIdx.x (line 02), not 2*threadIdx.x --
//     adjacent threads own adjacent locations, all in the first half of the
//     input array.
//   - stride starts at blockDim.x and is HALVED each iteration down to 1
//     (line 03) -- the opposite direction from Fig. 10.5's doubling stride.
//   - The active-thread test becomes `threadIdx.x < stride` (line 04): a
//     contiguous prefix of threads stays active, so entire low-numbered
//     warps remain fully active/fully inactive together for as long as
//     possible, and only the last few iterations (once the active-thread
//     count drops below 32) exhibit any divergence at all.
//
// §10.4's own worked numbers for a 256-element input: resource consumption
// drops from 864 (Fig. 10.5) to (4 + 2 + 1 + 5*1)*32 = 384 -- "almost half"
// -- while the useful-result count is unchanged at 255, so efficiency rises
// to 255/384 = 0.66, "almost double" Fig. 10.5's 0.30. §10.5 (file 03) adds
// that this same reassignment happens to *also* fix memory-access
// coalescing, as a bonus on top of the control-divergence fix demonstrated
// here.
//
// Like file 01, this kernel still overwrites `input` in place (still
// single-block, N <= 2048, still the "owner computes" discipline -- just a
// different owner assignment).

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

// ---------------------------------------------------------------------------
// §10.4, Fig. 10.8: reduced-control-divergence sum-reduction kernel.
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

// CPU reference, Fig. 10.1 (see file 01 for the double-accumulation
// rationale).
double reduction_cpu(const float *input, unsigned int n) {
    double sum = 0.0;
    for (unsigned int i = 0; i < n; ++i) {
        sum += input[i];
    }
    return sum;
}

// Same deterministic generator as file 01 (duplicated per this repo's
// self-contained-file convention).
std::vector<float> generateInput(unsigned int n) {
    std::vector<float> v(n);
    unsigned int state = 987654321u;
    for (unsigned int i = 0; i < n; ++i) {
        state = state * 1103515245u + 12345u;
        v[i] = static_cast<float>((state >> 8) & 0xFFFFFF) / static_cast<float>(0x1000000);
    }
    return v;
}

// Runs the convergent kernel once on an N-element input (N a power of two,
// N <= 2048). As in file 01, `input` is re-uploaded before both the
// untimed warm-up launch and the timed launch because the kernel mutates
// it in place.
float runConvergentReduction(const std::vector<float> &input_h, float *sum_out) {
    unsigned int n = static_cast<unsigned int>(input_h.size());
    size_t bytes = n * sizeof(float);

    float *input_d, *output_d;
    CUDA_CHECK(cudaMalloc((void **)&input_d, bytes));
    CUDA_CHECK(cudaMalloc((void **)&output_d, sizeof(float)));

    dim3 dimBlock(n / 2);
    dim3 dimGrid(1);

    CUDA_CHECK(cudaMemcpy(input_d, input_h.data(), bytes, cudaMemcpyHostToDevice));
    reduction_convergent_kernel<<<dimGrid, dimBlock>>>(input_d, output_d);  // warm-up
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(input_d, input_h.data(), bytes, cudaMemcpyHostToDevice));
    GpuTimer timer;
    timer.start();
    reduction_convergent_kernel<<<dimGrid, dimBlock>>>(input_d, output_d);
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
    float ms = runConvergentReduction(input_h, &gpuSum);

    bool ok = nearlyEqual(gpuSum, static_cast<float>(ref), 1e-2f);
    printf("N=%u: cpu=%.6f gpu=%.6f  %.4f ms  [%s]\n", n, ref, gpuSum, ms, ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    bool ok = true;
    // N must be a power of two <= 2048, same single-block constraint as
    // file 01, so the two kernels' timings below can be compared apples to
    // apples in the README.
    ok = runTestCase(128) && ok;
    ok = runTestCase(512) && ok;
    ok = runTestCase(2048) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
