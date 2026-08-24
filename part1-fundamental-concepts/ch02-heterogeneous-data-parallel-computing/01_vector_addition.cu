// Chapter 2: Heterogeneous data-parallel computing
// §2.3  A vector addition example       -- sequential (host) vecAdd, Fig. 2.4
// §2.4  Device global memory and data transfer
//       -- cudaMalloc / cudaMemcpy host code, Fig. 2.5 -> Fig. 2.8
// §2.5  Kernel functions and threading  -- vecAddKernel, Fig. 2.10
// §2.6  Calling kernel functions        -- ceil-division grid launch, Fig. 2.12/2.13
//
// This file follows the vector-addition running example that Chapter 2 builds
// up piece by piece: a plain sequential vecAdd (the "traditional" version),
// then the same computation reorganized so the addition itself runs on the
// device while the host is left with the role of an "outsourcing agent" that
// allocates device memory, ships data to the device, launches the kernel, and
// collects the result back.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

// ---------------------------------------------------------------------------
// §2.3, Fig. 2.4: traditional sequential vector addition (host code).
// Suffixed with "_h" per the book's convention for host-owned data.
// ---------------------------------------------------------------------------
void vecAdd_h(const float *A_h, const float *B_h, float *C_h, int n) {
    for (int i = 0; i < n; ++i) {
        C_h[i] = A_h[i] + B_h[i];
    }
}

// ---------------------------------------------------------------------------
// §2.5, Fig. 2.10: the vector addition kernel.
// Every thread computes a unique global index i from its block and thread
// coordinates and adds one pair of elements. The "if (i < n)" guard disables
// the extra threads generated when n is not an exact multiple of the block
// size (§2.5's discussion of vectors whose length isn't a multiple of 32/256).
// ---------------------------------------------------------------------------
__global__ void vecAddKernel(const float *A, const float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

// ---------------------------------------------------------------------------
// §2.4 + §2.6, Fig. 2.8 / Fig. 2.13: the revised vecAdd host function that
// outsources the computation to the device.
//   Part 1: cudaMalloc device buffers, cudaMemcpy A and B host -> device.
//   Part 2: launch vecAddKernel with a grid size computed via ceiling
//           division of n by the block size, as described in §2.6.
//   Part 3: cudaMemcpy C device -> host, cudaFree the device buffers.
// GPU kernel time is measured with GpuTimer around Part 2 only.
// ---------------------------------------------------------------------------
float vecAdd(const float *A_h, const float *B_h, float *C_h, int n) {
    float *A_d, *B_d, *C_d;
    int size = n * sizeof(float);

    // Part 1: allocate device global memory and copy inputs to the device.
    CUDA_CHECK(cudaMalloc((void **)&A_d, size));
    CUDA_CHECK(cudaMalloc((void **)&B_d, size));
    CUDA_CHECK(cudaMalloc((void **)&C_d, size));

    CUDA_CHECK(cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice));

    // Part 2: launch the kernel. §2.6: the number of blocks is the ceiling
    // division of n by the thread block size (256 threads/block, as in
    // Fig. 2.12/2.13) so the grid covers all n elements.
    const int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    GpuTimer timer;
    timer.start();
    vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d, C_d, n);
    CUDA_CHECK(cudaGetLastError());
    float ms = timer.stopAndGetMs();

    // Part 3: copy the result back to the host and free device memory.
    CUDA_CHECK(cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));

    return ms;
}

int main() {
    const int n = 1'000'000;

    std::vector<float> A_h(n), B_h(n), C_ref(n), C_h(n);
    for (int i = 0; i < n; ++i) {
        A_h[i] = static_cast<float>(i % 1000) * 0.5f;
        B_h[i] = static_cast<float>((i * 7) % 1000) * 0.25f;
    }

    // CPU reference (§2.3).
    vecAdd_h(A_h.data(), B_h.data(), C_ref.data(), n);

    // GPU version (§2.4-§2.6).
    float ms = vecAdd(A_h.data(), B_h.data(), C_h.data(), n);

    bool ok = true;
    for (int i = 0; i < n; ++i) {
        if (!nearlyEqual(C_h[i], C_ref[i])) {
            ok = false;
            fprintf(stderr, "Mismatch at i=%d: gpu=%f cpu=%f\n", i, C_h[i], C_ref[i]);
            break;
        }
    }

    printf("n = %d, threadsPerBlock = 256, blocksPerGrid = %d\n",
           n, (n + 255) / 256);
    printf("GPU vecAddKernel time: %.3f ms\n", ms);
    printf("%s\n", ok ? "PASS" : "FAIL");

    return ok ? 0 : 1;
}
