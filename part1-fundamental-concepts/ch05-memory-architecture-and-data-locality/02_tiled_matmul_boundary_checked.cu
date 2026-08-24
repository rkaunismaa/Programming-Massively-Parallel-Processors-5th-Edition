// Chapter 5: Memory architecture and data locality
// §5.5  Boundary checks -- matrixMulTiledBoundaryCheckedKernel, Fig. 5.13
//
// Same tiled algorithm as 01_tiled_matrix_multiplication.cu (§5.4, Fig. 5.9),
// generalized to square Width x Width matrices whose Width is *not* required
// to be a multiple of TILE_WIDTH. §5.5 walks through why this matters: in the
// last phase along either axis, some threads' tile-load indices run past the
// end of a row (landing on the wrong, but still in-bounds, element -- silent
// corruption) or past the end of the array entirely (out-of-bounds access).
// Crucially, §5.5 also shows this isn't confined to the "last phase": a
// thread that doesn't own a valid output element may still need to load a
// tile element that other threads in its block will consume (block1,1,
// phase 0 in Fig. 5.12), so every load needs its own bounds check rather
// than gating the whole thread on whether it owns a valid P element.
//
// The book's rule (§5.5): every global memory access gets its own bounds
// check.
//   - Loading M[Row][ph*TILE_WIDTH+tx]: valid iff Row < Width &&
//     (ph*TILE_WIDTH+tx) < Width. Otherwise store 0.0f into Mds -- a value
//     that contributes nothing to the dot product.
//   - Loading N[ph*TILE_WIDTH+ty][Col]: valid iff (ph*TILE_WIDTH+ty) < Width
//     && Col < Width. Otherwise store 0.0f into Nds.
//   - Writing P[Row][Col]: valid iff Row < Width && Col < Width.
//
// Tested here against square dimensions that are deliberately *not*
// multiples of TILE_WIDTH -- including one smaller than a single tile -- so
// every one of these boundary paths is actually exercised.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

#define TILE_WIDTH 32

// ---------------------------------------------------------------------------
// §5.5, Fig. 5.13: matrixMulTiledBoundaryCheckedKernel. Identical structure
// to §5.4's Fig. 5.9 kernel, with a bounds check guarding each of the two
// tile loads (zero-padding out-of-range elements) and the final P write.
// ---------------------------------------------------------------------------
__global__ void matrixMulTiledBoundaryCheckedKernel(const float *M, const float *N, float *P, int Width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0f;

    int numPhases = (Width + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int ph = 0; ph < numPhases; ++ph) {
        // Guard the M load: row must be in-bounds and the tile's column
        // offset must not run past the end of the row.
        if (Row < Width && (ph * TILE_WIDTH + tx) < Width) {
            Mds[ty][tx] = M[Row * Width + ph * TILE_WIDTH + tx];
        } else {
            Mds[ty][tx] = 0.0f;
        }

        // Guard the N load: the tile's row offset must not run past the end
        // of the column, and col must be in-bounds.
        if ((ph * TILE_WIDTH + ty) < Width && Col < Width) {
            Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];
        } else {
            Nds[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }

        __syncthreads();
    }

    // Guard the P write: only threads that own a valid output element store.
    if (Row < Width && Col < Width) {
        P[Row * Width + Col] = Pvalue;
    }
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

// Runs the boundary-checked kernel once (with a discarded warm-up launch
// first, per the chapter brief's timing-methodology note) and returns the
// timed kernel duration in ms. P_h receives the copied-back result.
float runTiledBoundaryChecked(const float *M_h, const float *N_h, float *P_h, int Width) {
    size_t size = static_cast<size_t>(Width) * Width * sizeof(float);

    float *M_d, *N_d, *P_d;
    CUDA_CHECK(cudaMalloc((void **)&M_d, size));
    CUDA_CHECK(cudaMalloc((void **)&N_d, size));
    CUDA_CHECK(cudaMalloc((void **)&P_d, size));
    CUDA_CHECK(cudaMemcpy(M_d, M_h, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(N_d, N_h, size, cudaMemcpyHostToDevice));

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid((Width + TILE_WIDTH - 1) / TILE_WIDTH, (Width + TILE_WIDTH - 1) / TILE_WIDTH, 1);

    // Warm-up launch (discarded) so PTX->SASS JIT cost isn't folded into the
    // timed measurement.
    matrixMulTiledBoundaryCheckedKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, Width);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    matrixMulTiledBoundaryCheckedKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, Width);
    CUDA_CHECK(cudaGetLastError());
    float ms = timer.stopAndGetMs();

    CUDA_CHECK(cudaMemcpy(P_h, P_d, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(M_d));
    CUDA_CHECK(cudaFree(N_d));
    CUDA_CHECK(cudaFree(P_d));

    return ms;
}

// Runs one Width test case: builds inputs, computes the CPU reference,
// launches the boundary-checked kernel, and checks agreement.
bool runTestCase(int Width) {
    size_t count = static_cast<size_t>(Width) * Width;
    std::vector<float> M_h(count), N_h(count), P_ref(count), P_h(count);

    for (size_t i = 0; i < count; ++i) {
        M_h[i] = static_cast<float>(i % 13) * 0.1f - 0.6f;
        N_h[i] = static_cast<float>(i % 7) * 0.2f - 0.6f;
    }

    matrixMul_h(M_h.data(), N_h.data(), P_ref.data(), Width);
    float ms = runTiledBoundaryChecked(M_h.data(), N_h.data(), P_h.data(), Width);

    bool ok = true;
    for (size_t i = 0; i < count; ++i) {
        if (!nearlyEqual(P_h[i], P_ref[i])) {
            ok = false;
            fprintf(stderr, "Mismatch at Width=%d i=%zu: gpu=%f cpu=%f\n", Width, i, P_h[i], P_ref[i]);
            break;
        }
    }

    dim3 dimGrid((Width + TILE_WIDTH - 1) / TILE_WIDTH, (Width + TILE_WIDTH - 1) / TILE_WIDTH, 1);
    int numPhases = (Width + TILE_WIDTH - 1) / TILE_WIDTH;
    printf("Width = %-4d (TILE_WIDTH=%d, %d phases, dimGrid=(%d,%d,1)): %.3f ms  [%s]\n",
           Width, TILE_WIDTH, numPhases, dimGrid.x, dimGrid.y, ms, ok ? "match" : "MISMATCH");

    return ok;
}

int main() {
    // Deliberately NOT multiples of TILE_WIDTH (32), so every phase --
    // including the last -- exercises the boundary checks:
    //   500 = 15*32 + 20   (several full phases, then a partial one)
    //    33 =  1*32 +  1   (one full phase, then a phase that is almost
    //                       entirely padding)
    //    15 =  0*32 + 15   (smaller than a single tile: the one and only
    //                       phase is all boundary-checked)
    const int widths[] = {500, 33, 15};

    bool ok = true;
    for (int w : widths) {
        ok = runTestCase(w) && ok;
    }

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
