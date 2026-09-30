// Chapter 5: Memory architecture and data locality
// §5.6  Impact of memory usage on occupancy -- dynamically sized shared
// memory (Fig. 5.14)
//
// 01_tiled_matrix_multiplication.cu and 02_tiled_matmul_boundary_checked.cu
// both hardwire TILE_WIDTH as a compile-time `#define`, exactly as Figs. 5.9
// and 5.13 do. §5.6 points out the consequence directly: "The kernels in
// Fig. 5.9 and Fig. 5.13 do not support any dynamic adjustment of shared
// memory usage by the host code ... Since the code contains
// `#define TILE_WIDTH 32` ... The kernel cannot easily adjust its shared
// memory usage at runtime without recompilation." Fig. 5.14 fixes this by
// merging `Mds`/`Nds` into one `extern __shared__` array whose size is
// supplied at launch time as the kernel-call's third `<<<...>>>` argument,
// with the two logical sections manually addressed inside a single
// linearized buffer.
//
// This sample implements that technique as a complete, runnable kernel
// (Fig. 5.14 itself only shows the declaration and the two pointer
// derivations, not a full kernel body) so that TILE_WIDTH becomes a
// *runtime* parameter -- the same kernel, uninstantiated and unrecompiled,
// is tested here at three different tile widths. We address the two
// sections of the merged buffer with plain element offsets (`Mds_Nds` and
// `Mds_Nds + tileWidth*tileWidth`, both already `float*`) rather than the
// byte-count parameter pair Fig. 5.14's own listing takes (whose interaction
// with the pointer casts on those lines is ambiguous from the printed
// text) -- this is an unambiguous way to do the same thing the figure is
// demonstrating: one dynamically sized buffer, manually split into two
// same-size logical sections.
//
// Otherwise this is the exact same algorithm as
// 01_tiled_matrix_multiplication.cu (§5.4, Fig. 5.9): square Width x Width
// matrices, Width assumed a multiple of tileWidth (Fig. 5.14 modifies
// Fig. 5.9's kernel, not the boundary-checked Fig. 5.13 one, so this file
// keeps that same simplifying assumption).

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

// ---------------------------------------------------------------------------
// §5.6, Fig. 5.14: matrixMulKernel with dynamically sized shared memory.
// tileWidth is a runtime parameter (not a compile-time #define), and must
// match the blockDim (tileWidth x tileWidth) and dynamic shared memory size
// (2 * tileWidth * tileWidth * sizeof(float)) the caller launches with.
// ---------------------------------------------------------------------------
__global__ void matrixMulTiledDynamicSharedKernel(const float *M, const float *N, float *P,
                                                    int Width, int tileWidth) {
    extern __shared__ float Mds_Nds[];
    float *Mds = Mds_Nds;
    float *Nds = Mds_Nds + tileWidth * tileWidth;

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int Row = by * tileWidth + ty;
    int Col = bx * tileWidth + tx;

    float Pvalue = 0.0f;

    for (int ph = 0; ph < Width / tileWidth; ++ph) {
        Mds[ty * tileWidth + tx] = M[Row * Width + ph * tileWidth + tx];
        Nds[ty * tileWidth + tx] = N[(ph * tileWidth + ty) * Width + Col];

        __syncthreads();

        for (int k = 0; k < tileWidth; ++k) {
            Pvalue += Mds[ty * tileWidth + k] * Nds[k * tileWidth + tx];
        }

        __syncthreads();
    }

    P[Row * Width + Col] = Pvalue;
}

// CPU reference: identical inner-product formula and loop order as §3.4/§5.4.
void matrixMul_h(const float *M, const float *N, float *P, int Width) {
    for (int row = 0; row < Width; ++row) {
        for (int col = 0; col < Width; ++col) {
            float Pvalue = 0.0f;
            for (int k = 0; k < Width; ++k) {
                Pvalue += M[row * Width + k] * N[k * Width + col];
            }
            P[row * Width + col] = Pvalue;
        }
    }
}

// Runs the dynamic-shared-memory kernel once at the given tileWidth and
// returns the timed kernel duration in ms; P_h receives the copied-back
// result. Width must be a multiple of tileWidth (see file header).
float runDynamicShared(const float *M_h, const float *N_h, float *P_h, int Width, int tileWidth) {
    size_t size = static_cast<size_t>(Width) * Width * sizeof(float);

    float *M_d, *N_d, *P_d;
    CUDA_CHECK(cudaMalloc((void **)&M_d, size));
    CUDA_CHECK(cudaMalloc((void **)&N_d, size));
    CUDA_CHECK(cudaMalloc((void **)&P_d, size));
    CUDA_CHECK(cudaMemcpy(M_d, M_h, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(N_d, N_h, size, cudaMemcpyHostToDevice));

    dim3 dimBlock(tileWidth, tileWidth, 1);
    dim3 dimGrid(Width / tileWidth, Width / tileWidth, 1);
    // The kernel-launch's third argument: bytes of dynamic shared memory,
    // covering both the Mds and Nds sections of the merged buffer.
    size_t sharedBytes = 2ull * tileWidth * tileWidth * sizeof(float);

    // Warm-up launch (discarded), as in every other timed sample in this repo.
    matrixMulTiledDynamicSharedKernel<<<dimGrid, dimBlock, sharedBytes>>>(M_d, N_d, P_d, Width,
                                                                           tileWidth);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    matrixMulTiledDynamicSharedKernel<<<dimGrid, dimBlock, sharedBytes>>>(M_d, N_d, P_d, Width,
                                                                           tileWidth);
    CUDA_CHECK(cudaGetLastError());
    float ms = timer.stopAndGetMs();

    CUDA_CHECK(cudaMemcpy(P_h, P_d, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(M_d));
    CUDA_CHECK(cudaFree(N_d));
    CUDA_CHECK(cudaFree(P_d));

    return ms;
}

int main() {
    // Chosen to be an exact multiple of every tile width tested below, so
    // the same inputs and the same CPU reference serve all three runs.
    const int Width = 256;

    size_t count = static_cast<size_t>(Width) * Width;
    std::vector<float> M_h(count), N_h(count), P_ref(count), P_h(count);
    for (size_t i = 0; i < count; ++i) {
        M_h[i] = static_cast<float>(i % 13) * 0.1f - 0.6f;
        N_h[i] = static_cast<float>(i % 7) * 0.2f - 0.6f;
    }

    printf("Computing CPU reference (Width=%d, %zu elements)...\n", Width, count);
    matrixMul_h(M_h.data(), N_h.data(), P_ref.data(), Width);

    // Same runtime kernel, three different tile widths -- no recompilation,
    // unlike 01_tiled_matrix_multiplication.cu's compile-time TILE_WIDTH.
    const int tileWidths[] = {8, 16, 32};

    bool ok = true;
    for (int tileWidth : tileWidths) {
        std::fill(P_h.begin(), P_h.end(), 0.0f);
        float ms = runDynamicShared(M_h.data(), N_h.data(), P_h.data(), Width, tileWidth);

        bool caseOk = true;
        for (size_t i = 0; i < count; ++i) {
            if (!nearlyEqual(P_h[i], P_ref[i])) {
                caseOk = false;
                fprintf(stderr, "Mismatch at tileWidth=%d i=%zu: gpu=%f cpu=%f\n", tileWidth, i,
                        P_h[i], P_ref[i]);
                break;
            }
        }
        ok = ok && caseOk;

        size_t sharedBytes = 2ull * tileWidth * tileWidth * sizeof(float);
        printf("tileWidth=%-3d (dynamic shared mem = %5zu B, dimBlock=(%d,%d,1)): %.3f ms  [%s]\n",
               tileWidth, sharedBytes, tileWidth, tileWidth, ms, caseOk ? "match" : "MISMATCH");
    }

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
