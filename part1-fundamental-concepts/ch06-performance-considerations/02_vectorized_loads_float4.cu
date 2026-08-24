// Chapter 6: Performance considerations
// §6.3  Vector loads and stores -- Fig. 6.12, a vector-addition kernel that
// uses float4 vector loads/stores, vs. the scalar vector-addition kernel of
// Fig. 2.10 (Chapter 2).
//
// The book's point: each thread of the scalar kernel issues one load/store
// instruction per 4 B element (two loads + one store per output element).
// Casting the array pointers to float4 before dereferencing (lines 03-04,
// 10 of Fig. 6.12) makes each thread issue vector load/store instructions
// that move 16 B (4 floats) in a single instruction, so two vector loads
// (x4, y4) replace eight scalar loads for the same data volume -- a 75%
// reduction in load-instruction count (as the text states directly below
// Fig. 6.12).
//
// §6.3 leaves the "n not divisible by 4" boundary condition as an exercise;
// this file sidesteps it by choosing n as an exact multiple of 4, so every
// thread of the vectorized kernel performs an in-bounds float4 access.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

// ---------------------------------------------------------------------------
// Fig. 2.10 (Chapter 2): scalar vector-addition kernel, reimplemented here
// (no cross-chapter include) as the baseline for comparison.
// ---------------------------------------------------------------------------
__global__ void vecAddScalarKernel(const float *x, const float *y, float *z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        z[i] = x[i] + y[i];
    }
}

// ---------------------------------------------------------------------------
// §6.3, Fig. 6.12: vectorized vector-addition kernel. Each thread handles
// one float4 (4 elements). The number of threads launched is n/4.
// ---------------------------------------------------------------------------
__global__ void vecAddVec4Kernel(const float *x, const float *y, float *z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // index of this thread's float4 chunk
    if (i * 4 < n) {
        float4 x4 = reinterpret_cast<const float4 *>(x)[i];
        float4 y4 = reinterpret_cast<const float4 *>(y)[i];
        float4 z4;
        z4.x = x4.x + y4.x;
        z4.y = x4.y + y4.y;
        z4.z = x4.z + y4.z;
        z4.w = x4.w + y4.w;
        reinterpret_cast<float4 *>(z)[i] = z4;
    }
}

void vecAdd_h(const float *x, const float *y, float *z, int n) {
    for (int i = 0; i < n; ++i) z[i] = x[i] + y[i];
}

int main() {
    const int n = 1 << 24;  // 16,777,216 -- exact multiple of 4
    if (n % 4 != 0) {
        fprintf(stderr, "n must be a multiple of 4 for this file\n");
        return 1;
    }

    size_t size = static_cast<size_t>(n) * sizeof(float);

    std::vector<float> x_h(n), y_h(n), z_ref(n), z_scalar_h(n), z_vec4_h(n);
    for (int i = 0; i < n; ++i) {
        x_h[i] = static_cast<float>(i % 1000) * 0.001f;
        y_h[i] = static_cast<float>((i * 7) % 1000) * 0.001f;
    }

    printf("Computing CPU reference (n=%d)...\n", n);
    vecAdd_h(x_h.data(), y_h.data(), z_ref.data(), n);

    float *x_d, *y_d, *z_d;
    CUDA_CHECK(cudaMalloc((void **)&x_d, size));
    CUDA_CHECK(cudaMalloc((void **)&y_d, size));
    CUDA_CHECK(cudaMalloc((void **)&z_d, size));
    CUDA_CHECK(cudaMemcpy(x_d, x_h.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(y_d, y_h.data(), size, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSizeScalar = (n + blockSize - 1) / blockSize;
    int gridSizeVec4 = (n / 4 + blockSize - 1) / blockSize;

    // Warm up both kernels once each (discarded) before timing.
    vecAddScalarKernel<<<gridSizeScalar, blockSize>>>(x_d, y_d, z_d, n);
    CUDA_CHECK(cudaGetLastError());
    vecAddVec4Kernel<<<gridSizeVec4, blockSize>>>(x_d, y_d, z_d, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    vecAddScalarKernel<<<gridSizeScalar, blockSize>>>(x_d, y_d, z_d, n);
    CUDA_CHECK(cudaGetLastError());
    float scalar_ms = timer.stopAndGetMs();
    CUDA_CHECK(cudaMemcpy(z_scalar_h.data(), z_d, size, cudaMemcpyDeviceToHost));

    timer.start();
    vecAddVec4Kernel<<<gridSizeVec4, blockSize>>>(x_d, y_d, z_d, n);
    CUDA_CHECK(cudaGetLastError());
    float vec4_ms = timer.stopAndGetMs();
    CUDA_CHECK(cudaMemcpy(z_vec4_h.data(), z_d, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(x_d));
    CUDA_CHECK(cudaFree(y_d));
    CUDA_CHECK(cudaFree(z_d));

    bool scalar_ok = true, vec4_ok = true;
    for (int i = 0; i < n; ++i) {
        if (scalar_ok && !nearlyEqual(z_scalar_h[i], z_ref[i])) {
            scalar_ok = false;
            fprintf(stderr, "Scalar mismatch at i=%d: gpu=%f cpu=%f\n", i, z_scalar_h[i], z_ref[i]);
        }
        if (vec4_ok && !nearlyEqual(z_vec4_h[i], z_ref[i])) {
            vec4_ok = false;
            fprintf(stderr, "float4 mismatch at i=%d: gpu=%f cpu=%f\n", i, z_vec4_h[i], z_ref[i]);
        }
        if (!scalar_ok && !vec4_ok) break;
    }

    printf("n = %d\n", n);
    printf("Scalar (Fig. 2.10, 1 thread/element)   kernel time: %.3f ms  [%s]\n",
           scalar_ms, scalar_ok ? "match" : "MISMATCH");
    printf("float4 (§6.3, Fig. 6.12, 1 thread/4 elements) kernel time: %.3f ms  [%s]\n",
           vec4_ms, vec4_ok ? "match" : "MISMATCH");
    printf("Speedup (scalar/float4): %.2fx\n", scalar_ms / vec4_ms);

    bool ok = scalar_ok && vec4_ok;
    printf("%s\n", ok ? "PASS" : "FAIL");

    return ok ? 0 : 1;
}
