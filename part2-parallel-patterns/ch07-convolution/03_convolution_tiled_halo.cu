// Chapter 7: Convolution and constant memory
// §7.5  Tiled convolution with halo cells -- convolution_tiled_2D_const_mem_kernel
//       (Fig. 7.12)
//
// Applies shared-memory tiling (as in Chapter 5's tiled matmul) to reduce
// DRAM traffic for N. Unlike tiled matmul, convolution's *input* tile is
// larger than its *output* tile: an output tile needs FILTER_RADIUS extra
// input elements ("halo cells") on every side to compute the output
// elements at its own edges (Fig. 7.11).
//
// This file uses the book's *first* thread organization (§7.5): the thread
// block is sized to match the *input* tile (IN_TILE_DIM x IN_TILE_DIM), so
// each thread loads exactly one input element into shared memory. This
// makes loading simple but means the block has more threads than the
// (smaller) output tile has elements, so FILTER_RADIUS layers of threads
// around the block's exterior are deactivated during the compute phase
// (Fig. 7.13) -- only threads with threadIdx in [FILTER_RADIUS,
// blockDim-FILTER_RADIUS) actually write an output element.
//
// IN_TILE_DIM = 32 (the max block width for a square block), so with
// FILTER_RADIUS = 2 the output tile is OUT_TILE_DIM = 32 - 2*2 = 28 --
// deliberately not a power of two, exactly the "second inefficiency" §7.5
// calls out about this design (addressed in §7.6, file 04).

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

#define FILTER_RADIUS 2
#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM - 2 * FILTER_RADIUS)

__constant__ float F[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];

// ---------------------------------------------------------------------------
// §7.5, Fig. 7.12: convolution_tiled_2D_const_mem_kernel.
//
//   - col/row (lines 06-07 of Fig. 7.12): the N index this thread loads,
//     offset by -FILTER_RADIUS from the output-tile origin so that the
//     block's IN_TILE_DIM x IN_TILE_DIM footprint covers the output tile
//     plus its halo on every side.
//   - N_s load (lines 09-15): every thread loads one element into shared
//     memory; out-of-range (ghost cell) loads store 0.0f instead.
//   - __syncthreads() (line 15): the whole input tile must be staged before
//     any thread starts consuming it.
//   - Compute (lines 17-29): only the interior FILTER_RADIUS..blockDim-1-
//     FILTER_RADIUS threads are "active" and write an output element, using
//     tileCol/tileRow as the upper-left corner of their filter-sized patch
//     within N_s (Fig. 7.13). Because col/row already carry the same
//     -FILTER_RADIUS offset used for loading, they double as the correct
//     output P index for active threads without any extra arithmetic.
// ---------------------------------------------------------------------------
__global__ void convolution_tiled_2D_const_mem_kernel(const float *N, float *P, int width, int height) {
    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;

    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];

    // Load the input tile (including halo) into shared memory; ghost cells
    // (outside the valid N range) are zero-filled.
    if (row >= 0 && row < height && col >= 0 && col < width) {
        N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // Only the interior threads (the output tile) compute and write P.
    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;

    if (col >= 0 && col < width && row >= 0 && row < height) {
        if (threadIdx.x >= FILTER_RADIUS && threadIdx.x < IN_TILE_DIM - FILTER_RADIUS && threadIdx.y >= FILTER_RADIUS &&
            threadIdx.y < IN_TILE_DIM - FILTER_RADIUS) {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; ++fRow) {
                for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; ++fCol) {
                    Pvalue += F[fRow][fCol] * N_s[tileRow + fRow][tileCol + fCol];
                }
            }
            P[row * width + col] = Pvalue;
        }
    }
}

// CPU reference: same weighted-sum-with-zero-ghost-cells formula as
// 02_convolution_constant_memory.cu, reading the filter from a flat host
// array (row-major).
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

// Runs the tiled-halo kernel once (with a discarded warm-up launch first)
// and returns the timed kernel duration in ms. Grid dimensions are computed
// from OUT_TILE_DIM (the output tile size), while each block itself spans
// IN_TILE_DIM x IN_TILE_DIM threads.
float runTiledHaloConvolution(const float *N_h, const float *F_h, float *P_h, int width, int height) {
    size_t nSize = static_cast<size_t>(width) * height * sizeof(float);
    size_t fSize = static_cast<size_t>(2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1) * sizeof(float);

    float *N_d, *P_d;
    CUDA_CHECK(cudaMalloc((void **)&N_d, nSize));
    CUDA_CHECK(cudaMalloc((void **)&P_d, nSize));
    CUDA_CHECK(cudaMemcpy(N_d, N_h, nSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(F, F_h, fSize));

    dim3 dimBlock(IN_TILE_DIM, IN_TILE_DIM, 1);
    dim3 dimGrid((width + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM, 1);

    // Warm-up launch (discarded) so PTX->SASS JIT cost isn't folded into the
    // timed measurement.
    convolution_tiled_2D_const_mem_kernel<<<dimGrid, dimBlock>>>(N_d, P_d, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    convolution_tiled_2D_const_mem_kernel<<<dimGrid, dimBlock>>>(N_d, P_d, width, height);
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
    float ms = runTiledHaloConvolution(N_h.data(), F_h.data(), P_h.data(), width, height);

    bool ok = true;
    for (size_t i = 0; i < nCount; ++i) {
        if (!nearlyEqual(P_h[i], P_ref[i])) {
            ok = false;
            fprintf(stderr, "Mismatch at width=%d height=%d i=%zu: gpu=%f cpu=%f\n", width, height, i, P_h[i],
                    P_ref[i]);
            break;
        }
    }

    dim3 dimGrid((width + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM, 1);
    printf("width=%-4d height=%-4d (IN_TILE_DIM=%d, OUT_TILE_DIM=%d, dimGrid=(%d,%d,1)): %.3f ms  [%s]\n", width,
           height, IN_TILE_DIM, OUT_TILE_DIM, dimGrid.x, dimGrid.y, ms, ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    bool ok = true;
    // Mix of sizes that are and aren't multiples of OUT_TILE_DIM (28), so
    // both partial output tiles and ghost-cell handling on all four edges
    // are exercised.
    ok = runTestCase(67, 51) && ok;
    ok = runTestCase(56, 56) && ok;  // exact multiple of OUT_TILE_DIM
    ok = runTestCase(300, 200) && ok;
    ok = runTestCase(1024, 1024) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
