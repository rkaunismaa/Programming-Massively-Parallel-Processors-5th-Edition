// Chapter 8: Stencil computation
// §8.5  Thread coarsening (Fig. 8.10)
//
// File 02's cubic tiling is limited to small tiles (IN_TILE_DIM=8, i.e.
// 512 threads) because a cubic thread block of side t has t^3 threads and
// the hardware caps blocks at 1024 threads. §8.5's fix: coarsen each
// thread's work from "one output grid point" to "a whole column of output
// grid points along z", so the thread BLOCK only needs t^2 threads (one
// per x-y plane grid point) no matter how many z-planes it sweeps. This is
// a genuinely different technique from file 02's tiling, not a renamed
// copy of it: file 02 launches one thread per 3D grid point and one block
// per (small) 3D tile; this file launches one thread per 2D (x,y) column
// and has each thread loop, in software, over every z-plane its block is
// responsible for -- fewer threads and fewer blocks per output point
// processed, with the z-sweep itself replacing the missing z-extent of
// the thread grid.
//
// Because the block is now only 2D (t^2, not t^3), t can be much larger:
// IN_TILE_DIM = 32 gives a 1024-thread block (the hardware max), which
// §8.5 shows raises arithmetic intensity from 0.96 FLOP/B (t=8 cubic tile,
// file 02) to 1.52 FLOP/B -- much closer to the 1.625 FLOP/B ideal.
//
// At any instant only 3 z-planes of the input tile matter for the plane
// currently being computed: the previous plane (z-1 neighbor), the
// current plane (whose 4 in-plane neighbors + center are used), and the
// next plane (z+1 neighbor). Rather than keep the whole t^3 input tile in
// shared memory (as file 02 does for its small t), this kernel keeps only
// those 3 t^2 planes in shared memory (inPrev_s/inCurr_s/inNext_s) and
// slides them forward by one z-plane after each iteration -- 3*t^2
// elements instead of t^3, letting shared-memory consumption stay
// reasonable (12 KB per block at t=32, per §8.5) even though t itself
// grew 4x versus file 02.

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

#define IN_TILE_DIM 32
#define OUT_TILE_DIM (IN_TILE_DIM - 2)

// ---------------------------------------------------------------------------
// §8.5, Fig. 8.10: 3D 7-point stencil sweep with thread coarsening in z.
//
//   - iStart/j/k: the block covers output z-planes [iStart, iStart +
//     OUT_TILE_DIM) (coarsening range); j/k are the (offset by -1, for the
//     halo) x-y grid point this thread's whole column is responsible for.
//   - Initial load (inPrev_s <- plane iStart-1, inCurr_s <- plane iStart):
//     before the sweep can compute the first output plane, the block
//     needs the previous and current input planes staged.
//   - Per-iteration loop over i in [iStart, iStart+OUT_TILE_DIM):
//       * load inNext_s <- plane i+1 (the only new plane each iteration
//         needs, since the other two were already staged/computed from
//         the previous iteration or the initial load);
//       * __syncthreads() so every thread sees the fully-loaded 3 planes;
//       * interior threads compute out[i,j,k] from inCurr_s's 4 in-plane
//         neighbors plus inPrev_s/inNext_s at the same (j,k);
//       * __syncthreads(), then every thread slides the planes forward:
//         inPrev_s <- inCurr_s, inCurr_s <- inNext_s, ready for the next
//         iteration's plane i+2 load into inNext_s.
// ---------------------------------------------------------------------------
__global__ void stencil_coarsened_kernel(const float *in, float *out, unsigned int N) {
    int iStart = blockIdx.z * OUT_TILE_DIM;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

    __shared__ float inPrev_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inNext_s[IN_TILE_DIM][IN_TILE_DIM];

    if (iStart - 1 >= 0 && iStart - 1 < (int)N && j >= 0 && j < (int)N && k >= 0 && k < (int)N) {
        inPrev_s[threadIdx.y][threadIdx.x] = in[(iStart - 1) * N * N + j * N + k];
    }
    if (iStart >= 0 && iStart < (int)N && j >= 0 && j < (int)N && k >= 0 && k < (int)N) {
        inCurr_s[threadIdx.y][threadIdx.x] = in[iStart * N * N + j * N + k];
    }
    __syncthreads();

    for (int i = iStart; i < iStart + OUT_TILE_DIM; ++i) {
        if (i + 1 >= 0 && i + 1 < (int)N && j >= 0 && j < (int)N && k >= 0 && k < (int)N) {
            inNext_s[threadIdx.y][threadIdx.x] = in[(i + 1) * N * N + j * N + k];
        }
        __syncthreads();

        if (i >= 1 && i < (int)N - 1 && j >= 1 && j < (int)N - 1 && k >= 1 && k < (int)N - 1) {
            if (threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 && threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1) {
                out[i * N * N + j * N + k] =
                    C0 * inCurr_s[threadIdx.y][threadIdx.x] + C1 * inCurr_s[threadIdx.y][threadIdx.x - 1] +
                    C2 * inCurr_s[threadIdx.y][threadIdx.x + 1] + C3 * inCurr_s[threadIdx.y - 1][threadIdx.x] +
                    C4 * inCurr_s[threadIdx.y + 1][threadIdx.x] + C5 * inPrev_s[threadIdx.y][threadIdx.x] +
                    C6 * inNext_s[threadIdx.y][threadIdx.x];
            }
        }
        __syncthreads();

        // Slide the 3-plane window forward by one z-plane for the next
        // iteration: current becomes previous, next becomes current.
        inPrev_s[threadIdx.y][threadIdx.x] = inCurr_s[threadIdx.y][threadIdx.x];
        inCurr_s[threadIdx.y][threadIdx.x] = inNext_s[threadIdx.y][threadIdx.x];
        __syncthreads();
    }
}

// CPU reference: identical to 01_stencil_naive.cu.
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

// Runs the thread-coarsened kernel once (with a discarded warm-up launch
// first) and returns the timed kernel duration in ms. Each block spans
// IN_TILE_DIM x IN_TILE_DIM threads (a single x-y plane's worth) and
// sweeps OUT_TILE_DIM output z-planes internally, so gridDim.z is
// ceil(N / OUT_TILE_DIM) rather than one block per output point.
float runCoarsenedStencil(const float *in_h, float *out_h, unsigned int N) {
    size_t count = static_cast<size_t>(N) * N * N;
    size_t bytes = count * sizeof(float);

    float *in_d, *out_d;
    CUDA_CHECK(cudaMalloc((void **)&in_d, bytes));
    CUDA_CHECK(cudaMalloc((void **)&out_d, bytes));
    CUDA_CHECK(cudaMemcpy(in_d, in_h, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(out_d, out_h, bytes, cudaMemcpyHostToDevice));  // sentinel-filled

    dim3 dimBlock(IN_TILE_DIM, IN_TILE_DIM, 1);
    dim3 dimGrid((N + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                 (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM);

    stencil_coarsened_kernel<<<dimGrid, dimBlock>>>(in_d, out_d, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out_d, out_h, bytes, cudaMemcpyHostToDevice));  // reset to sentinel

    GpuTimer timer;
    timer.start();
    stencil_coarsened_kernel<<<dimGrid, dimBlock>>>(in_d, out_d, N);
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
    float ms = runCoarsenedStencil(in_h.data(), out_h.data(), N);

    bool ok = true;
    for (size_t idx = 0; idx < count; ++idx) {
        if (!nearlyEqual(out_h[idx], out_ref[idx])) {
            ok = false;
            fprintf(stderr, "Mismatch at N=%u idx=%zu: gpu=%f cpu=%f\n", N, idx, out_h[idx], out_ref[idx]);
            break;
        }
    }

    printf("N=%-4u (IN_TILE_DIM=%d, OUT_TILE_DIM=%d, z-coarsened): %.3f ms  [%s]\n", N, IN_TILE_DIM, OUT_TILE_DIM, ms,
           ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    bool ok = true;
    // Mix of sizes that are and aren't multiples of OUT_TILE_DIM (30), so
    // both partial coarsening ranges and ghost-cell handling on all faces
    // are exercised.
    ok = runTestCase(34) && ok;
    ok = runTestCase(62) && ok;  // 60 interior points = exact multiple of 30
    ok = runTestCase(94) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
