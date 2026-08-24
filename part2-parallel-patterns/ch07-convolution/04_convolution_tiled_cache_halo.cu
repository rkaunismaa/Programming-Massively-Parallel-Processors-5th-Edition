// Chapter 7: Convolution and constant memory
// §7.6  Tiled convolution using caches for halo cells --
//       convolution_cached_tiled_2D_const_mem_kernel (Fig. 7.15)
//
// §7.5's kernel (file 03) explicitly stages the halo into shared memory,
// which forces the block/input-tile dimension to be larger than the output
// tile and leaves the output tile size an awkward non-power-of-two.
//
// §7.6's insight: a block's halo cells are exactly the *interior* elements
// of its neighboring blocks' input tiles. By the time a block needs them,
// there's a good chance they're already sitting in the L2 cache from a
// neighboring block's own loads -- so there is no need to explicitly stage
// them into shared memory at all. This kernel loads *only* the interior
// (non-halo) elements of the tile into shared memory N_s, and for halo
// accesses simply reads N directly from global memory (relying on the L2
// cache), still skipping true ghost cells (outside the array) as 0.
//
// Because the shared-memory tile now holds only the interior elements,
// input tile size == output tile size == block size == TILE_DIM, which can
// be a power of two (32) -- this is the "subtle advantage" §7.6 calls out
// over §7.5's kernel.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

#define FILTER_RADIUS 2
#define TILE_DIM 32

__constant__ float F[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];

// ---------------------------------------------------------------------------
// §7.6, Fig. 7.15: convolution_cached_tiled_2D_const_mem_kernel.
//
//   - row/col (lines 4-5): the output/input element this thread is
//     responsible for -- input tile and output tile share the same
//     dimension and index here, unlike file 03.
//   - N_s load (lines 7-11): each thread loads exactly the element with its
//     own row/col (no halo, no -FILTER_RADIUS offset), so the only
//     condition needed is the ordinary "does this tile extend past the
//     array" boundary check -- no ghost-cell zero-fill is possible here
//     since there's no halo being loaded.
//   - __syncthreads(): the interior tile must be fully staged before use.
//   - Compute (lines 17-27): for each filter tap, if the needed N element
//     falls within this block's own tile (lines 17-20) it's read from
//     shared memory N_s; otherwise it's a halo cell, and if it's also a
//     true ghost cell (outside the array, lines 24-27) it's skipped as 0,
//     else it's read directly from global memory N (served from L2 in the
//     common case).
// ---------------------------------------------------------------------------
__global__ void convolution_cached_tiled_2D_const_mem_kernel(const float *N, float *P, int width, int height) {
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    int row = blockIdx.y * TILE_DIM + threadIdx.y;

    __shared__ float N_s[TILE_DIM][TILE_DIM];

    // Load only the internal tile elements into shared memory; no halo.
    if (row < height && col < width) {
        N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    if (row < height && col < width) {
        float Pvalue = 0.0f;
        for (int fRow = -FILTER_RADIUS; fRow <= FILTER_RADIUS; ++fRow) {
            for (int fCol = -FILTER_RADIUS; fCol <= FILTER_RADIUS; ++fCol) {
                int tRow = static_cast<int>(threadIdx.y) + fRow;
                int tCol = static_cast<int>(threadIdx.x) + fCol;
                if (tRow >= 0 && tRow < TILE_DIM && tCol >= 0 && tCol < TILE_DIM) {
                    // Interior of this block's own tile: served from shared memory.
                    Pvalue += F[fRow + FILTER_RADIUS][fCol + FILTER_RADIUS] * N_s[tRow][tCol];
                } else {
                    // Halo cell: read directly from global memory (L2-cached in the
                    // common case), unless it's also a true ghost cell.
                    int inRow = row + fRow;
                    int inCol = col + fCol;
                    if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                        Pvalue += F[fRow + FILTER_RADIUS][fCol + FILTER_RADIUS] * N[inRow * width + inCol];
                    }
                }
            }
        }
        P[row * width + col] = Pvalue;
    }
}

// CPU reference: same weighted-sum-with-zero-ghost-cells formula as the
// other files, reading the filter from a flat host array (row-major).
void convolution2D_h(const float *N, const float *F_h, float *P, int width, int height) {
    for (int outRow = 0; outRow < height; ++outRow) {
        for (int outCol = 0; outCol < width; ++outCol) {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; ++fRow) {
                for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; ++fCol) {
                    int inRow = outRow - FILTER_RADIUS + fRow;
                    int inCol = outCol - FILTER_RADIUS + fCol;
                    if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                        Pvalue += F_h[fRow * (2 * FILTER_RADIUS + 1) + fCol] * N[inRow * width + inCol];
                    }
                }
            }
            P[outRow * width + outCol] = Pvalue;
        }
    }
}

// Runs the cache-halo kernel once (with a discarded warm-up launch first)
// and returns the timed kernel duration in ms. Here block size == input
// tile size == output tile size == TILE_DIM in both dimensions.
float runCacheHaloConvolution(const float *N_h, const float *F_h, float *P_h, int width, int height) {
    size_t nSize = static_cast<size_t>(width) * height * sizeof(float);
    size_t fSize = static_cast<size_t>(2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1) * sizeof(float);

    float *N_d, *P_d;
    CUDA_CHECK(cudaMalloc((void **)&N_d, nSize));
    CUDA_CHECK(cudaMalloc((void **)&P_d, nSize));
    CUDA_CHECK(cudaMemcpy(N_d, N_h, nSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(F, F_h, fSize));

    dim3 dimBlock(TILE_DIM, TILE_DIM, 1);
    dim3 dimGrid((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM, 1);

    // Warm-up launch (discarded) so PTX->SASS JIT cost isn't folded into the
    // timed measurement.
    convolution_cached_tiled_2D_const_mem_kernel<<<dimGrid, dimBlock>>>(N_d, P_d, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    convolution_cached_tiled_2D_const_mem_kernel<<<dimGrid, dimBlock>>>(N_d, P_d, width, height);
    CUDA_CHECK(cudaGetLastError());
    float ms = timer.stopAndGetMs();

    CUDA_CHECK(cudaMemcpy(P_h, P_d, nSize, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(N_d));
    CUDA_CHECK(cudaFree(P_d));

    return ms;
}

// Runs one (width, height) test case with a fixed FILTER_RADIUS-2 filter:
// builds N and F, computes the CPU reference, launches the kernel, and
// checks agreement.
bool runTestCase(int width, int height) {
    size_t nCount = static_cast<size_t>(width) * height;
    size_t fCount = static_cast<size_t>(2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1);

    std::vector<float> N_h(nCount), P_ref(nCount), P_h(nCount);
    std::vector<float> F_h(fCount);

    for (size_t i = 0; i < nCount; ++i) {
        N_h[i] = static_cast<float>(i % 13) * 0.1f - 0.6f;
    }
    for (size_t i = 0; i < fCount; ++i) {
        F_h[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.1f;
    }

    convolution2D_h(N_h.data(), F_h.data(), P_ref.data(), width, height);
    float ms = runCacheHaloConvolution(N_h.data(), F_h.data(), P_h.data(), width, height);

    bool ok = true;
    for (size_t i = 0; i < nCount; ++i) {
        if (!nearlyEqual(P_h[i], P_ref[i])) {
            ok = false;
            fprintf(stderr, "Mismatch at width=%d height=%d i=%zu: gpu=%f cpu=%f\n", width, height, i, P_h[i],
                    P_ref[i]);
            break;
        }
    }

    dim3 dimGrid((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM, 1);
    printf("width=%-4d height=%-4d (TILE_DIM=%d, dimGrid=(%d,%d,1)): %.3f ms  [%s]\n", width, height, TILE_DIM,
           dimGrid.x, dimGrid.y, ms, ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    bool ok = true;
    // Mix of sizes that are and aren't multiples of TILE_DIM (32), so both
    // partial tiles and ghost-cell handling on all four edges are
    // exercised.
    ok = runTestCase(67, 51) && ok;
    ok = runTestCase(64, 64) && ok;  // exact multiple of TILE_DIM
    ok = runTestCase(300, 200) && ok;
    ok = runTestCase(1024, 1024) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
