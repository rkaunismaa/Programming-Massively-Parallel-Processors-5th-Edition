// Chapter 10: Reduction
// §10.6  Reducing global memory accesses (Fig. 10.9, Fig. 10.10)
//
// Even the coalescing-fixed kernel of §10.5/file 02 still round-trips every
// partial sum through GLOBAL memory: each iteration writes partial sums out
// to `input` in DRAM/L2 and rereads them next iteration. §10.6's fix is to
// keep the partial sums in shared memory instead, which "has much lower
// latency and higher bandwidth than the last-level cache and the global
// memory" (Fig. 10.9).
//
// Fig. 10.10's kernel:
//   - Each thread loads and adds its TWO original elements directly from
//     global memory ONCE (line 04: input_s[t] = input[t] + input[t +
//     blockDim.x]) and writes the partial sum into shared memory -- this
//     does the work of the first tree level "for free" outside the loop,
//     which is why the loop below starts at blockDim.x/2 rather than
//     blockDim.x.
//   - __syncthreads() moves to the TOP of the loop (rather than the bottom,
//     as in files 01-03) so it synchronizes both the initial shared-memory
//     load above and every subsequent iteration's shared-memory write.
//   - All remaining iterations read and write shared memory only (line 08),
//     using the exact same contiguous-active-thread strategy as Fig. 10.8
//     (`t < stride`, stride halving from blockDim.x/2 down to 1) so control
//     divergence and (now-moot, since it's on-chip) memory divergence stay
//     minimized.
//   - Thread 0 writes the final sum from input_s[0] to *output (lines
//     11-13).
//
// §10.6 derives the exact global-memory-request count: for an N-element
// reduction, DRAM traffic is now just the N-element initial load plus the
// 1-element final write, i.e. N+1 accesses, and because the two reads on
// line 04 are themselves coalesced, that's only (N/32)+1 global memory
// REQUESTS. For the running 256-element example: 8+1 = 9 requests, "a 4x
// improvement" over the 36 requests the book states directly (no
// re-derivation shown in §10.6's text) the coalesced-but-global-memory
// kernel of file 02 (Fig. 10.8) would issue for the same input.
//
// Unlike files 01-03, this kernel does NOT modify the input array (it only
// reads from it into shared memory) -- §10.6 calls this out as "useful if
// the original values of the array are needed for some other computation."

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

// ---------------------------------------------------------------------------
// §10.6, Fig. 10.10: shared-memory sum-reduction kernel. `input_s` is sized
// to blockDim.x floats via dynamic shared memory so this one kernel can be
// exercised at multiple block sizes in runTestCase below.
// ---------------------------------------------------------------------------
__global__ void reduction_shared_kernel(const float *input, float *output) {
    extern __shared__ float input_s[];
    unsigned int t = threadIdx.x;

    input_s[t] = input[t] + input[t + blockDim.x];

    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            input_s[t] += input_s[t + stride];
        }
    }

    if (t == 0) {
        *output = input_s[0];
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

// Runs the shared-memory kernel once on an N-element input (N a power of
// two, N <= 2048). Unlike files 01-03, `input` is read-only on the device,
// so a single upload before the warm-up launch suffices for both launches.
float runSharedReduction(const std::vector<float> &input_h, float *sum_out) {
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

    reduction_shared_kernel<<<dimGrid, dimBlock, shmemBytes>>>(input_d, output_d);  // warm-up
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    reduction_shared_kernel<<<dimGrid, dimBlock, shmemBytes>>>(input_d, output_d);
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
    float ms = runSharedReduction(input_h, &gpuSum);

    bool ok = nearlyEqual(gpuSum, static_cast<float>(ref), 1e-2f);
    printf("N=%u: cpu=%.6f gpu=%.6f  %.4f ms  [%s]\n", n, ref, gpuSum, ms, ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    bool ok = true;
    // Same single-block power-of-two constraint as files 01-03.
    ok = runTestCase(128) && ok;
    ok = runTestCase(512) && ok;
    ok = runTestCase(2048) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
