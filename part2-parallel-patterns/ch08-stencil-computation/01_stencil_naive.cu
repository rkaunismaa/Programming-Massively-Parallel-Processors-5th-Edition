// Chapter 8: Stencil computation
// §8.2  Parallel stencil -- a basic kernel (Fig. 8.6)
//
// A stencil sweep applies a fixed geometric pattern of weights to every
// grid point of a structured n-dimensional grid, producing an output value
// per point from that point's current value and its immediate neighbors'
// values. This file implements the book's basic (untiled) 3D 7-point
// stencil sweep: the order-1 stencil of Fig. 8.3(c) -- the center grid
// point plus one neighbor on each side along x, y, and z (7 points, 13
// FLOPs per output point: 7 multiplies + 6 adds, §8.3).
//
// Per §8.2's simplifying assumption (Fig. 8.5), the outermost layer of the
// grid stores fixed boundary conditions and is never written by the
// sweep -- only the (N-2)^3 interior points are computed. This file's test
// harness mirrors that directly: both the CPU reference and the GPU output
// buffer are pre-filled with a sentinel value before the sweep runs, so
// comparing the *entire* N^3 buffer afterward verifies both that the
// interior was computed correctly and that the boundary layer was left
// untouched (matching in[] rather than being overwritten by either side).
//
// Each thread is assigned to exactly one output grid point (Fig. 8.6,
// lines 02-04) and reads its 7 stencil inputs directly from global memory
// -- no shared memory, no coarsening. This is the memory-bound baseline
// (arithmetic intensity 0.41 FLOP/B per §8.3) that 02/03/04 progressively
// optimize.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

#define SENTINEL (-12345.0f)

// 3D 7-point stencil coefficients (order-1: one neighbor per axis side).
// Hard-coded per §8.2's remark that stencil coefficients "may be
// hard-coded in the code, or ... placed in constant memory depending on
// the desired level of flexibility" -- their values come from whatever
// differential equation is being solved and are not the subject of this
// chapter's optimizations, so a simple hard-coded scheme is used here.
// C1/C2 weight the x-neighbors (k-1/k+1), C3/C4 the y-neighbors (j-1/j+1),
// C5/C6 the z-neighbors (i-1/i+1).
#define C0 1.500f
#define C1 0.100f
#define C2 0.200f
#define C3 0.300f
#define C4 0.400f
#define C5 0.500f
#define C6 0.600f

// ---------------------------------------------------------------------------
// §8.2, Fig. 8.6: basic stencil sweep kernel.
//
//   - i/j/k (lines 02-04 of Fig. 8.6): the 3D grid point this thread is
//     responsible for, taken directly from blockIdx/blockDim/threadIdx
//     with no offset (unlike the tiled kernels, there is no halo to load).
//   - The interior guard (line 05-ish, "if boundary check"): only threads
//     whose grid point falls in [1, N-2] on every axis compute an output
//     value; boundary-layer threads do nothing, leaving out[] untouched
//     there per Fig. 8.5's simplifying assumption.
//   - The 7-point sum (c0..c6): center point (c0) plus one neighbor on
//     each side along x, y, z -- all read straight from global memory.
// ---------------------------------------------------------------------------
__global__ void stencil_kernel(const float *in, float *out, unsigned int N) {
    unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
        out[i * N * N + j * N + k] = C0 * in[i * N * N + j * N + k] + C1 * in[i * N * N + j * N + (k - 1)] +
                                      C2 * in[i * N * N + j * N + (k + 1)] + C3 * in[i * N * N + (j - 1) * N + k] +
                                      C4 * in[i * N * N + (j + 1) * N + k] + C5 * in[(i - 1) * N * N + j * N + k] +
                                      C6 * in[(i + 1) * N * N + j * N + k];
    }
}

// CPU reference: identical 7-point weighted sum over the interior; the
// boundary layer is left as whatever sentinel the caller pre-filled out[]
// with, exactly like the kernel leaves it as whatever was already in the
// device output buffer.
void stencil_cpu(const float *in, float *out, unsigned int N) {
    for (unsigned int i = 1; i < N - 1; ++i) {
        for (unsigned int j = 1; j < N - 1; ++j) {
            for (unsigned int k = 1; k < N - 1; ++k) {
                out[i * N * N + j * N + k] = C0 * in[i * N * N + j * N + k] + C1 * in[i * N * N + j * N + (k - 1)] +
                                              C2 * in[i * N * N + j * N + (k + 1)] +
                                              C3 * in[i * N * N + (j - 1) * N + k] +
                                              C4 * in[i * N * N + (j + 1) * N + k] +
                                              C5 * in[(i - 1) * N * N + j * N + k] +
                                              C6 * in[(i + 1) * N * N + j * N + k];
            }
        }
    }
}

// Runs the basic stencil kernel once (with a discarded warm-up launch
// first) and returns the timed kernel duration in ms.
float runNaiveStencil(const float *in_h, float *out_h, unsigned int N) {
    size_t count = static_cast<size_t>(N) * N * N;
    size_t bytes = count * sizeof(float);

    float *in_d, *out_d;
    CUDA_CHECK(cudaMalloc((void **)&in_d, bytes));
    CUDA_CHECK(cudaMalloc((void **)&out_d, bytes));
    CUDA_CHECK(cudaMemcpy(in_d, in_h, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(out_d, out_h, bytes, cudaMemcpyHostToDevice));  // sentinel-filled

    dim3 dimBlock(8, 8, 8);
    dim3 dimGrid((N + 7) / 8, (N + 7) / 8, (N + 7) / 8);

    // Warm-up launch (discarded) so PTX->SASS JIT cost isn't folded into
    // the timed measurement.
    stencil_kernel<<<dimGrid, dimBlock>>>(in_d, out_d, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out_d, out_h, bytes, cudaMemcpyHostToDevice));  // reset to sentinel

    GpuTimer timer;
    timer.start();
    stencil_kernel<<<dimGrid, dimBlock>>>(in_d, out_d, N);
    CUDA_CHECK(cudaGetLastError());
    float ms = timer.stopAndGetMs();

    CUDA_CHECK(cudaMemcpy(out_h, out_d, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(in_d));
    CUDA_CHECK(cudaFree(out_d));

    return ms;
}

// Runs one N x N x N test case: builds a deterministic input grid, computes
// the CPU reference (interior swept, boundary left as sentinel), launches
// the kernel, and checks full-buffer agreement (interior values AND
// untouched boundary).
bool runTestCase(unsigned int N) {
    size_t count = static_cast<size_t>(N) * N * N;
    std::vector<float> in_h(count), out_ref(count, SENTINEL), out_h(count, SENTINEL);

    for (unsigned int i = 0; i < N; ++i) {
        for (unsigned int j = 0; j < N; ++j) {
            for (unsigned int k = 0; k < N; ++k) {
                in_h[i * N * N + j * N + k] = static_cast<float>((i * 31 + j * 17 + k * 7) % 100) * 0.031f - 1.5f;
            }
        }
    }

    stencil_cpu(in_h.data(), out_ref.data(), N);
    float ms = runNaiveStencil(in_h.data(), out_h.data(), N);

    bool ok = true;
    for (size_t idx = 0; idx < count; ++idx) {
        if (!nearlyEqual(out_h[idx], out_ref[idx])) {
            ok = false;
            fprintf(stderr, "Mismatch at N=%u idx=%zu: gpu=%f cpu=%f\n", N, idx, out_h[idx], out_ref[idx]);
            break;
        }
    }

    printf("N=%-4u (%u^3=%zu points): %.3f ms  [%s]\n", N, N, count, ms, ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    bool ok = true;
    // Mix of sizes that are and aren't multiples of the 8x8x8 block, so
    // partial blocks at the grid's far edges are exercised too.
    ok = runTestCase(24) && ok;
    ok = runTestCase(40) && ok;  // exact multiple of block dim (8)
    ok = runTestCase(66) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
