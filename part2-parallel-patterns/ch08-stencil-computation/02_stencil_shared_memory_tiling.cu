// Chapter 8: Stencil computation
// §8.4  Shared-memory tiling for stencil sweep (Fig. 8.8)
//
// Applies the same shared-memory tiling idea used for tiled convolution
// (Chapter 7, Fig. 7.12) to the 3D 7-point stencil sweep: the thread block
// is sized to match the *input* tile (IN_TILE_DIM^3), each thread loads
// exactly one input element into shared memory, and only the interior
// threads (whose grid point falls inside the smaller output tile) compute
// and write an output value.
//
// Unlike convolution's input tile, the stencil's input tile does not
// include corner grid points (§8.4, Fig. 8.7 vs Fig. 7.11): the 7-point
// stencil only ever touches face neighbors, never diagonals. Ghost cells
// therefore never need to be zero-filled here (contrast Chapter 7's tiled
// kernel) -- any shared-memory slot that a boundary thread failed to load
// is simply never read, because the compute phase only reads shared-memory
// positions that a valid load guaranteed were populated.
//
// §8.3 established that the 1024-thread block-size limit makes 3D cubic
// tiles small in practice: an 8x8x8 block (512 threads) is the practical
// limit, per the book's own analysis of this kernel. IN_TILE_DIM = 8 is
// used here for that reason -- deliberately small, which is exactly the
// motivation for thread coarsening in §8.5 (file 03) and register tiling
// in §8.6 (file 04): a small cubic tile has ~58% of its input tile as
// halo (§8.4), so the arithmetic intensity of *this* kernel (0.96 FLOP/B
// at t=8) falls well short of the 1.625 FLOP/B ideal.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

#define SENTINEL (-12345.0f)

#define C0 1.500f
#define C1 0.100f
#define C2 0.200f
#define C3 0.300f
#define C4 0.400f
#define C5 0.500f
#define C6 0.600f

#define IN_TILE_DIM 8
#define OUT_TILE_DIM (IN_TILE_DIM - 2)

// ---------------------------------------------------------------------------
// §8.4, Fig. 8.8: 3D 7-point stencil sweep with shared-memory tiling.
//
//   - i/j/k (lines 02-04 of Fig. 8.8): the grid point this thread loads,
//     offset by -1 (the stencil's order) from the output-tile origin so
//     the block's IN_TILE_DIM^3 footprint covers the output tile plus its
//     1-cell halo on every side.
//   - in_s load (lines 05-07): every thread loads one element into shared
//     memory, guarded against out-of-range (ghost cell) loads -- but
//     unlike convolution, out-of-range slots are simply left unwritten
//     (never zero-filled), since the 7-point stencil's compute phase never
//     reads a shared-memory position that wasn't validly loaded.
//   - __syncthreads() (line 09): the whole input tile must be staged
//     before any thread starts consuming it.
//   - Compute (lines 10-21): only interior threads (both in the valid grid
//     interior AND away from the block's IN_TILE_DIM halo layer) write an
//     output element, reading all 7 stencil points from shared memory
//     instead of global memory.
// ---------------------------------------------------------------------------
__global__ void stencil_tiled_kernel(const float *in, float *out, unsigned int N) {
    int i = blockIdx.z * OUT_TILE_DIM + threadIdx.z - 1;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

    __shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];

    if (i >= 0 && i < (int)N && j >= 0 && j < (int)N && k >= 0 && k < (int)N) {
        in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[i * N * N + j * N + k];
    }
    __syncthreads();

    if (i >= 1 && i < (int)N - 1 && j >= 1 && j < (int)N - 1 && k >= 1 && k < (int)N - 1) {
        if (threadIdx.z >= 1 && threadIdx.z < IN_TILE_DIM - 1 && threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 &&
            threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1) {
            out[i * N * N + j * N + k] =
                C0 * in_s[threadIdx.z][threadIdx.y][threadIdx.x] + C1 * in_s[threadIdx.z][threadIdx.y][threadIdx.x - 1] +
                C2 * in_s[threadIdx.z][threadIdx.y][threadIdx.x + 1] + C3 * in_s[threadIdx.z][threadIdx.y - 1][threadIdx.x] +
                C4 * in_s[threadIdx.z][threadIdx.y + 1][threadIdx.x] + C5 * in_s[threadIdx.z - 1][threadIdx.y][threadIdx.x] +
                C6 * in_s[threadIdx.z + 1][threadIdx.y][threadIdx.x];
        }
    }
}

// CPU reference: identical to 01_stencil_naive.cu -- same 7-point weighted
// sum over the interior, boundary left as sentinel.
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

// Runs the shared-memory-tiled kernel once (with a discarded warm-up
// launch first) and returns the timed kernel duration in ms. Grid
// dimensions are computed from OUT_TILE_DIM, while each block spans
// IN_TILE_DIM^3 threads.
float runTiledStencil(const float *in_h, float *out_h, unsigned int N) {
    size_t count = static_cast<size_t>(N) * N * N;
    size_t bytes = count * sizeof(float);

    float *in_d, *out_d;
    CUDA_CHECK(cudaMalloc((void **)&in_d, bytes));
    CUDA_CHECK(cudaMalloc((void **)&out_d, bytes));
    CUDA_CHECK(cudaMemcpy(in_d, in_h, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(out_d, out_h, bytes, cudaMemcpyHostToDevice));  // sentinel-filled

    dim3 dimBlock(IN_TILE_DIM, IN_TILE_DIM, IN_TILE_DIM);
    dim3 dimGrid((N + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                 (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

    stencil_tiled_kernel<<<dimGrid, dimBlock>>>(in_d, out_d, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out_d, out_h, bytes, cudaMemcpyHostToDevice));  // reset to sentinel

    GpuTimer timer;
    timer.start();
    stencil_tiled_kernel<<<dimGrid, dimBlock>>>(in_d, out_d, N);
    CUDA_CHECK(cudaGetLastError());
    float ms = timer.stopAndGetMs();

    CUDA_CHECK(cudaMemcpy(out_h, out_d, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(in_d));
    CUDA_CHECK(cudaFree(out_d));

    return ms;
}

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
    float ms = runTiledStencil(in_h.data(), out_h.data(), N);

    bool ok = true;
    for (size_t idx = 0; idx < count; ++idx) {
        if (!nearlyEqual(out_h[idx], out_ref[idx])) {
            ok = false;
            fprintf(stderr, "Mismatch at N=%u idx=%zu: gpu=%f cpu=%f\n", N, idx, out_h[idx], out_ref[idx]);
            break;
        }
    }

    printf("N=%-4u (IN_TILE_DIM=%d, OUT_TILE_DIM=%d): %.3f ms  [%s]\n", N, IN_TILE_DIM, OUT_TILE_DIM, ms,
           ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    bool ok = true;
    // Mix of sizes that are and aren't multiples of OUT_TILE_DIM (6), so
    // both partial output tiles and ghost-cell handling on all faces are
    // exercised.
    ok = runTestCase(20) && ok;
    ok = runTestCase(32) && ok;  // 30 interior points = exact multiple of 6
    ok = runTestCase(50) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
