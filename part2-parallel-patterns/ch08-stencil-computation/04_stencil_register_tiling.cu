// Chapter 8: Stencil computation
// §8.6  Register tiling (Fig. 8.12)
//
// File 03's coarsening kernel keeps all 3 active z-planes (prev/curr/next)
// in shared memory. §8.6 observes that this is wasteful for a stencil
// whose neighbors lie only along the x, y, z axes (true of every stencil
// in Fig. 8.3, including the 3D 7-point stencil used throughout this
// chapter): inPrev_s[y][x] and inNext_s[y][x] are each read by exactly ONE
// thread -- the thread whose own (y,x) column they belong to -- during the
// z-neighbor term of that single thread's own output computation. Only
// inCurr_s is genuinely *shared*, because computing an output point also
// needs its 4 in-plane (x-1/x+1/y-1/y+1) neighbors, which belong to other
// threads' columns.
//
// Since inPrev/inNext are single-thread-private data, §8.6's optimization
// is to stop shared-memory-broadcasting them at all: they become plain
// per-thread register variables (`float inPrev`, `float inNext`) instead
// of `__shared__` arrays. inCurr, however, is kept in BOTH a register
// (for this thread's own center-point term) AND the shared array
// inCurr_s (so neighboring threads can still read this thread's current
// value for their in-plane terms) -- that duplication is exactly what
// register tiling means here: the same value lives in a register for its
// owning thread's fast, private access, while still being visible to
// other threads via shared memory when they need it as a neighbor. This
// is a genuinely different technique from file 03, not a renamed copy:
// file 03 keeps ALL THREE z-planes in shared memory and reads all 7
// stencil terms from shared arrays; this file keeps only ONE z-plane
// (inCurr_s) in shared memory and folds the other two z-terms into
// straight register reads (c5*inPrev / c6*inNext), replacing 2 of the 7
// shared-memory loads per output point with register accesses that have
// far lower latency and higher bandwidth (§8.6).
//
// The result: shared-memory consumption per block drops to 1/3 of file
// 03's (one t^2 plane instead of three, e.g. 4 KB vs 12 KB at t=32), paid
// for by 3 extra registers per thread (3072 more registers per 32x32
// block, per §8.6). Global memory traffic and overall data reuse are
// unchanged versus file 03 -- register tiling only redistributes where
// the *on-chip* reuse happens, so this optimization does not by itself
// reduce DRAM bandwidth demand any further than file 03 already achieved.

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
// §8.6, Fig. 8.12: 3D 7-point stencil sweep with thread coarsening AND
// register tiling in z.
//
//   - inPrev/inNext (registers) replace file 03's inPrev_s/inNext_s
//     shared-memory arrays entirely -- each thread holds only its own
//     (j,k) column's previous/next z-value.
//   - inCurr_s (shared) is retained, but each thread ALSO keeps its own
//     current-plane value in the register `inCurr`; the register is used
//     for this thread's own center-point term (c0*inCurr) while inCurr_s
//     is used only when reading a NEIGHBORING thread's in-plane value
//     (c1..c4 terms).
//   - Per-iteration loop over i in [iStart, iStart+OUT_TILE_DIM): load the
//     next plane's value into the register `inNext`; sync; compute using
//     inPrev/inCurr/inNext (registers) plus inCurr_s (shared, for in-plane
//     neighbors); sync; then slide the register window forward
//     (inPrev <- inCurr, inCurr <- inNext) and publish the new inCurr into
//     inCurr_s for the next iteration's neighbors to read.
// ---------------------------------------------------------------------------
__global__ void stencil_register_tiled_kernel(const float *in, float *out, unsigned int N) {
    int iStart = blockIdx.z * OUT_TILE_DIM;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

    float inPrev = 0.0f;
    float inCurr = 0.0f;
    float inNext = 0.0f;
    __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];

    if (iStart - 1 >= 0 && iStart - 1 < (int)N && j >= 0 && j < (int)N && k >= 0 && k < (int)N) {
        inPrev = in[(iStart - 1) * N * N + j * N + k];
    }
    if (iStart >= 0 && iStart < (int)N && j >= 0 && j < (int)N && k >= 0 && k < (int)N) {
        inCurr = in[iStart * N * N + j * N + k];
        inCurr_s[threadIdx.y][threadIdx.x] = inCurr;
    }
    __syncthreads();

    for (int i = iStart; i < iStart + OUT_TILE_DIM; ++i) {
        if (i + 1 >= 0 && i + 1 < (int)N && j >= 0 && j < (int)N && k >= 0 && k < (int)N) {
            inNext = in[(i + 1) * N * N + j * N + k];
        }
        __syncthreads();

        if (i >= 1 && i < (int)N - 1 && j >= 1 && j < (int)N - 1 && k >= 1 && k < (int)N - 1) {
            if (threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 && threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1) {
                out[i * N * N + j * N + k] =
                    C0 * inCurr + C1 * inCurr_s[threadIdx.y][threadIdx.x - 1] +
                    C2 * inCurr_s[threadIdx.y][threadIdx.x + 1] + C3 * inCurr_s[threadIdx.y - 1][threadIdx.x] +
                    C4 * inCurr_s[threadIdx.y + 1][threadIdx.x] + C5 * inPrev + C6 * inNext;
            }
        }
        __syncthreads();

        // Slide the register window forward by one z-plane, and publish
        // the new current plane into shared memory for the next
        // iteration's in-plane neighbor reads.
        inPrev = inCurr;
        inCurr = inNext;
        inCurr_s[threadIdx.y][threadIdx.x] = inNext;
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

// Runs the register-tiled kernel once (with a discarded warm-up launch
// first) and returns the timed kernel duration in ms. Grid/block shape is
// identical to file 03's -- only the kernel body's use of registers vs.
// shared memory for inPrev/inNext differs.
float runRegisterTiledStencil(const float *in_h, float *out_h, unsigned int N) {
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

    stencil_register_tiled_kernel<<<dimGrid, dimBlock>>>(in_d, out_d, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out_d, out_h, bytes, cudaMemcpyHostToDevice));  // reset to sentinel

    GpuTimer timer;
    timer.start();
    stencil_register_tiled_kernel<<<dimGrid, dimBlock>>>(in_d, out_d, N);
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
    float ms = runRegisterTiledStencil(in_h.data(), out_h.data(), N);

    bool ok = true;
    for (size_t idx = 0; idx < count; ++idx) {
        if (!nearlyEqual(out_h[idx], out_ref[idx])) {
            ok = false;
            fprintf(stderr, "Mismatch at N=%u idx=%zu: gpu=%f cpu=%f\n", N, idx, out_h[idx], out_ref[idx]);
            break;
        }
    }

    printf("N=%-4u (IN_TILE_DIM=%d, OUT_TILE_DIM=%d, z-coarsened+register-tiled): %.3f ms  [%s]\n", N, IN_TILE_DIM,
           OUT_TILE_DIM, ms, ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    bool ok = true;
    ok = runTestCase(34) && ok;
    ok = runTestCase(62) && ok;  // 60 interior points = exact multiple of 30
    ok = runTestCase(94) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
