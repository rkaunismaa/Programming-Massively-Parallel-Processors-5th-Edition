// Chapter 4: Compute architecture and scheduling
// §4.5  Control divergence -- divergence at a for-loop (Fig. 4.10)
//
// §4.5 covers two distinct divergence shapes: an if-else (Fig. 4.9, modeled
// by 01_control_divergence_demo.cu) and a for-loop whose trip count is
// itself thread/data-dependent (Fig. 4.10): "each thread executes a
// different number of loop iterations, which vary between four and eight
// ... For the fifth iteration, some threads execute A, while others are
// inactive because they have completed their iterations." Unlike Fig.
// 4.9's if-else, the book does not present a "fixed" version of this
// kernel -- and there isn't a simple one. An if-else's two branches are
// two genuinely separate code paths, so replacing the branch with an
// arithmetic select really does halve the number of passes a warp takes
// (that is what 01_control_divergence_demo.cu measures). A data-dependent
// *loop trip count* is different: SIMT execution already re-issues the
// loop body, masking off lanes that have finished, until the *slowest*
// lane in the warp is done -- so a warp's cost is bounded by the maximum
// N among its 32 lanes no matter how the loop is written. There is no
// branchless rewrite that removes this; the only real fix is to restructure
// the *work assignment* so that lanes sharing a warp have similar trip
// counts, which is a substantially different (and heavier) technique than
// anything else in this chapter.
//
// This sample therefore reproduces Fig. 4.10's exact shape and numeric
// range ("vary between four and eight": nVals[i] = 4 + (i % 5), so every
// warp's 32 lanes span multiple distinct N values since 32 is not a
// multiple of 5) and measures its cost against two non-divergent
// *baseline* kernels that each run a fixed, uniform number of iterations
// for every thread -- not alternative ways to compute the same per-element
// answer, but reference points for how expensive the divergence is:
//   - uniformMaxKernel: every thread always runs N_MAX iterations. Since
//     every 32-lane warp here contains at least one element with N ==
//     N_MAX (see nVals above), this is what the hardware is already doing
//     for divergentLoopKernel -- so their times are expected to be close.
//   - uniformAvgKernel: every thread always runs round(mean(N)) == 6
//     iterations -- the amount of useful work there would be *if* it could
//     be spread across lanes with no idle cycles. Since 6 < N_MAX == 8,
//     this is expected to run measurably faster than the other two,
//     quantifying the fraction of each warp's cycles spent on lanes that
//     have already finished (the "inactive" lanes Fig. 4.10 depicts).
//
// Correctness (PASS/FAIL) is judged solely by whether divergentLoopKernel
// agrees with a CPU reference that applies exactly N multiplies per
// element -- the two uniform kernels compute a deliberately different,
// simplified quantity (fixed iteration counts, not per-element N) and are
// for timing only, so they are not checked against that reference. All
// three kernels are warmed up once (discarding results) before timing, as
// in 01_control_divergence_demo.cu, to keep one-time PTX->SASS JIT cost
// from leaking into whichever kernel is timed first.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

// Fig. 4.10's own numeric range: "vary between four and eight."
constexpr int N_MIN = 4;
constexpr int N_MAX = 8;
// round(mean(4,5,6,7,8)) == round(6.0) == 6.
constexpr int N_AVG = 6;

// Multiplies executed per loop iteration. Purely to make the divergence
// cost measurable (see file header); does not affect which elements
// diverge or by how many iterations.
constexpr int WORK_PER_ITER = 500;

constexpr float FACTOR = 1.0000001f;

// ---------------------------------------------------------------------------
// §4.5, Fig. 4.10: for-loop whose trip count N is data/thread-dependent.
// nVals[i] cycles 4,5,6,7,8 with period 5; since blockDim.x (256) and the
// warp size (32) are both multiples of 5's complement in the worst way (32
// is not a multiple of 5), every warp's 32 lanes span multiple distinct N
// values, so every warp takes multiple passes at the loop, one lane group
// at a time, exactly as Fig. 4.10 illustrates.
// ---------------------------------------------------------------------------
__global__ void divergentLoopKernel(const float *in, float *out, const int *nVals, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        int N = nVals[i];
        for (int outer = 0; outer < N; ++outer) {
            for (int k = 0; k < WORK_PER_ITER; ++k) {
                x *= FACTOR;
            }
        }
        out[i] = x;
    }
}

// ---------------------------------------------------------------------------
// Baseline: every thread runs the same N_MAX iterations, unconditionally --
// no thread-index-dependent trip count, no divergence. Since every warp in
// divergentLoopKernel already contains an N == N_MAX lane and so is already
// bounded by N_MAX passes, this baseline's time is expected to closely
// track divergentLoopKernel's, not beat it.
// ---------------------------------------------------------------------------
__global__ void uniformMaxKernel(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        for (int outer = 0; outer < N_MAX; ++outer) {
            for (int k = 0; k < WORK_PER_ITER; ++k) {
                x *= FACTOR;
            }
        }
        out[i] = x;
    }
}

// ---------------------------------------------------------------------------
// Baseline: every thread runs N_AVG (< N_MAX) iterations, unconditionally --
// the amount of work that would be done if the same total work were spread
// perfectly evenly with no idle lanes. The gap between this and the two
// kernels above is the SIMD efficiency lost to stragglers.
// ---------------------------------------------------------------------------
__global__ void uniformAvgKernel(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        for (int outer = 0; outer < N_AVG; ++outer) {
            for (int k = 0; k < WORK_PER_ITER; ++k) {
                x *= FACTOR;
            }
        }
        out[i] = x;
    }
}

// CPU reference for divergentLoopKernel: exactly N iterations of
// WORK_PER_ITER multiplies by FACTOR.
void referenceHost(const float *in, const int *nVals, float *out, int n) {
    for (int i = 0; i < n; ++i) {
        float x = in[i];
        int N = nVals[i];
        for (int outer = 0; outer < N; ++outer) {
            for (int k = 0; k < WORK_PER_ITER; ++k) {
                x *= FACTOR;
            }
        }
        out[i] = x;
    }
}

int main() {
    const int n = 1 << 20;  // ~1M elements
    const int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;

    std::vector<float> in_h(n);
    std::vector<int> n_h(n);
    for (int i = 0; i < n; ++i) {
        in_h[i] = 1.0f + static_cast<float>(i % 97) * 0.01f;
        n_h[i] = N_MIN + (i % 5);  // cycles 4,5,6,7,8 -- Fig. 4.10's own range
    }

    std::vector<float> ref_h(n);
    referenceHost(in_h.data(), n_h.data(), ref_h.data(), n);

    float *in_d, *outDivergent_d, *outUniformMax_d, *outUniformAvg_d;
    int *n_d;
    CUDA_CHECK(cudaMalloc((void **)&in_d, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&outDivergent_d, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&outUniformMax_d, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&outUniformAvg_d, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&n_d, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(in_d, in_h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(n_d, n_h.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    dim3 dimBlock(blockSize);
    dim3 dimGrid(gridSize);

    // Warm-up launches (results discarded): absorb one-time PTX->SASS JIT
    // cost so it doesn't leak into whichever kernel is timed first below
    // (see 01_control_divergence_demo.cu for the full rationale).
    divergentLoopKernel<<<dimGrid, dimBlock>>>(in_d, outDivergent_d, n_d, n);
    CUDA_CHECK(cudaGetLastError());
    uniformMaxKernel<<<dimGrid, dimBlock>>>(in_d, outUniformMax_d, n);
    CUDA_CHECK(cudaGetLastError());
    uniformAvgKernel<<<dimGrid, dimBlock>>>(in_d, outUniformAvg_d, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;

    timer.start();
    divergentLoopKernel<<<dimGrid, dimBlock>>>(in_d, outDivergent_d, n_d, n);
    CUDA_CHECK(cudaGetLastError());
    float msDivergent = timer.stopAndGetMs();

    timer.start();
    uniformMaxKernel<<<dimGrid, dimBlock>>>(in_d, outUniformMax_d, n);
    CUDA_CHECK(cudaGetLastError());
    float msUniformMax = timer.stopAndGetMs();

    timer.start();
    uniformAvgKernel<<<dimGrid, dimBlock>>>(in_d, outUniformAvg_d, n);
    CUDA_CHECK(cudaGetLastError());
    float msUniformAvg = timer.stopAndGetMs();

    std::vector<float> outDivergent_h(n);
    CUDA_CHECK(cudaMemcpy(outDivergent_h.data(), outDivergent_d, n * sizeof(float),
                           cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(in_d));
    CUDA_CHECK(cudaFree(outDivergent_d));
    CUDA_CHECK(cudaFree(outUniformMax_d));
    CUDA_CHECK(cudaFree(outUniformAvg_d));
    CUDA_CHECK(cudaFree(n_d));

    bool ok = true;
    for (int i = 0; i < n; ++i) {
        if (!nearlyEqual(outDivergent_h[i], ref_h[i])) {
            ok = false;
            fprintf(stderr, "Mismatch at %d: divergent=%f ref=%f\n", i, outDivergent_h[i],
                    ref_h[i]);
            break;
        }
    }

    printf("n=%d, N in [%d,%d] (avg %d), WORK_PER_ITER=%d, dimBlock=(%d), dimGrid=(%d)\n", n,
           N_MIN, N_MAX, N_AVG, WORK_PER_ITER, blockSize, gridSize);
    printf("GPU divergentLoopKernel time: %.3f ms  (Fig. 4.10: data-dependent trip count)\n",
           msDivergent);
    printf("GPU uniformMaxKernel    time: %.3f ms  (every thread always runs N_MAX=%d)\n",
           msUniformMax, N_MAX);
    printf("GPU uniformAvgKernel    time: %.3f ms  (every thread always runs N_AVG=%d)\n",
           msUniformAvg, N_AVG);
    printf("(Only divergentLoopKernel is checked against a per-element CPU reference for\n"
           " PASS/FAIL; the uniform kernels compute a different, fixed-iteration-count\n"
           " quantity and are timing baselines only, illustrating that a warp already pays\n"
           " for its slowest lane's trip count -- so divergentLoopKernel's time tracks\n"
           " uniformMaxKernel's, not the smaller uniformAvgKernel's.)\n");
    printf("%s\n", ok ? "PASS" : "FAIL");

    return ok ? 0 : 1;
}
