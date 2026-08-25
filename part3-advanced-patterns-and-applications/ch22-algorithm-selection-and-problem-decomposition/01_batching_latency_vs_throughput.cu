// Chapter 22: Algorithm selection, problem decomposition, and problem
// formulation
// §22.5 "Batching: latency vs. throughput".
//
// This chapter is design-methodology focused: §22.1-22.4 review algorithm
// selection and problem decomposition trade-offs entirely by revisiting
// kernels from earlier chapters in prose (no new code or pseudocode is
// given), and §22.3's Amdahl's-law discussion is a fully worked arithmetic
// example (1/(5%+95%/100) = 17x, etc.) with no algorithm to implement.
// §22.5 is the one place in the chapter that describes a concrete,
// self-contained computational comparison -- it is NOT accompanied by a
// code listing or figure in the book (there is no "Fig. 22.x" for it), so
// this file is a faithful reconstruction of the described comparison using
// standard techniques from earlier chapters, not a transcription of book
// code. See the chapter-22 README for the reasoning behind treating this
// as in-scope despite the absence of a listing.
//
// The book's example (quoted from §22.5): a QKV projection is a
// vector-matrix multiplication for one query. "Batching queries from
// different users allows these queries to be processed by the transformer
// layers together to increase the arithmetic intensity of the attention
// heads. ... batching allows the QKV projection vector-matrix
// multiplication operations to be converted into matrix multiplications.
// Although a matrix multiplication takes more time to execute than a
// vector-matrix multiplication, it takes much shorter time than the sum of
// the execution time for all the vector-matrix multiplications in the
// batch. ... individual queries experience longer QKV projection latency,
// but the whole group of queries finish with shorter total latency. Thus,
// the throughput of QKV projection is higher with batching (total number
// of queries / latency for matrix multiplication) than without batching
// (total number of queries / total latency for all matrix-vector
// multiplications)."
//
// This file implements exactly that comparison for a linear projection
// Y = X W^T (X: B queries x d features, W: d x d projection weights,
// stored [out_features][in_features] as in real transformer weight
// layout -- the standard convention this book's Chapter 20 attention
// samples also assume for Q/K/V projections):
//
//   - UNBATCHED: one matvecKernel launch per query (B separate kernel
//     launches, each a naive one-thread-per-output-feature vector-matrix
//     multiply with no cross-query reuse of W). This models processing
//     each query "as it arrives" -- a single query's own launch finishes
//     quickly (short individual latency), but W is re-read from global
//     memory independently by every launch, and B kernel-launch overheads
//     accumulate, so the TOTAL latency to finish all B queries is large.
//
//   - BATCHED: one tiled shared-memory matmul kernel launch (Chapter 5's
//     tiling technique) computing all B outputs at once. Tiles of W are
//     loaded into shared memory once and reused across a whole tile of
//     queries, raising arithmetic intensity exactly as the book describes.
//     Any individual query's result is not ready until this single launch
//     completes, so its own latency is the whole batch's latency (longer
//     than the unbatched case's single-query latency) -- but the TOTAL
//     latency for all B queries, and therefore throughput
//     (queries / total latency), is far better than the unbatched path.
//
// CPU reference: double-precision reference computed independently of
// either kernel's decomposition, used to check both GPU paths for
// correctness. The two GPU paths are also cross-checked against each
// other since they compute the same mathematical result via different
// decompositions.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

#define D_FEATURES 256   // query/output feature dimension d
#define NUM_QUERIES 2048 // batch size B (number of independent queries)
#define TILE_WIDTH 16

// ---------------------------------------------------------------------------
// UNBATCHED path: naive vector-matrix multiply, one thread per output
// feature, no shared-memory reuse across queries. Launched once per query.
// y[r] = sum_k W[r*d+k] * x[k]
// ---------------------------------------------------------------------------
__global__ void matvecKernel(const float *W, const float *x, float *y, int d) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= d) return;
    float sum = 0.0f;
    for (int k = 0; k < d; ++k) {
        sum += W[r * d + k] * x[k];
    }
    y[r] = sum;
}

// ---------------------------------------------------------------------------
// BATCHED path: tiled shared-memory matmul computing Y = X * W^T in one
// launch (Chapter 5's tiling technique applied to an A*B^T access pattern).
// X: [B][d] row-major (query-major), W: [d][d] row-major
// ([out_features][in_features]), Y: [B][d] row-major.
// Y[q][r] = sum_k X[q*d+k] * W[r*d+k]
// ---------------------------------------------------------------------------
__global__ void batchedProjectionKernel(const float *X, const float *W, float *Y, int B, int d) {
    __shared__ float Xs[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Ws[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x, ty = threadIdx.y;
    int q = blockIdx.y * TILE_WIDTH + ty;   // query index (row of X / Y)
    int r = blockIdx.x * TILE_WIDTH + tx;   // output feature index (row of W / col of Y)

    float sum = 0.0f;
    int numTiles = (d + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int t = 0; t < numTiles; ++t) {
        int kX = t * TILE_WIDTH + tx;
        Xs[ty][tx] = (q < B && kX < d) ? X[q * d + kX] : 0.0f;
        int kW = t * TILE_WIDTH + ty;
        Ws[ty][tx] = (r < d && kW < d) ? W[r * d + kW] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += Xs[ty][k] * Ws[k][tx];
        }
        __syncthreads();
    }

    if (q < B && r < d) {
        Y[q * d + r] = sum;
    }
}

// ---------------------------------------------------------------------------
// CPU reference: double-precision Y[q][r] = sum_k X[q][k] * W[r][k].
// ---------------------------------------------------------------------------
static void cpuReference(const std::vector<float> &X, const std::vector<float> &W,
                          std::vector<float> &Y, int B, int d) {
    Y.assign(static_cast<size_t>(B) * d, 0.0f);
    for (int q = 0; q < B; ++q) {
        for (int r = 0; r < d; ++r) {
            double sum = 0.0;
            for (int k = 0; k < d; ++k) {
                sum += static_cast<double>(X[static_cast<size_t>(q) * d + k]) *
                       static_cast<double>(W[static_cast<size_t>(r) * d + k]);
            }
            Y[static_cast<size_t>(q) * d + r] = static_cast<float>(sum);
        }
    }
}

static std::vector<float> makeRandom(size_t n, unsigned int seed, float lo, float hi) {
    std::vector<float> v(n);
    unsigned int state = seed;
    auto nextRand = [&]() -> float {
        state = state * 1103515245u + 12345u;
        return static_cast<float>((state >> 8) & 0xFFFFFFu) / static_cast<float>(0xFFFFFFu);
    };
    for (size_t i = 0; i < n; ++i) {
        v[i] = lo + nextRand() * (hi - lo);
    }
    return v;
}

static bool checkClose(const std::vector<float> &a, const std::vector<float> &b, float eps, float *maxDiff) {
    bool ok = true;
    float md = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > md) md = diff;
        if (!nearlyEqual(a[i], b[i], eps)) ok = false;
    }
    if (maxDiff) *maxDiff = md;
    return ok;
}

int main() {
    const int d = D_FEATURES;
    const int B = NUM_QUERIES;
    printf("Batching latency vs. throughput (Sec.22.5): d=%d features, B=%d queries\n", d, B);

    std::vector<float> W = makeRandom(static_cast<size_t>(d) * d, 1u, -0.05f, 0.05f);
    std::vector<float> X = makeRandom(static_cast<size_t>(B) * d, 2u, -1.0f, 1.0f);

    std::vector<float> Yref;
    cpuReference(X, W, Yref, B, d);

    float *W_d, *X_d, *Yunb_d, *Ybat_d;
    size_t wBytes = static_cast<size_t>(d) * d * sizeof(float);
    size_t xBytes = static_cast<size_t>(B) * d * sizeof(float);
    CUDA_CHECK(cudaMalloc(&W_d, wBytes));
    CUDA_CHECK(cudaMalloc(&X_d, xBytes));
    CUDA_CHECK(cudaMalloc(&Yunb_d, xBytes));
    CUDA_CHECK(cudaMalloc(&Ybat_d, xBytes));
    CUDA_CHECK(cudaMemcpy(W_d, W.data(), wBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(X_d, X.data(), xBytes, cudaMemcpyHostToDevice));

    // -----------------------------------------------------------------
    // UNBATCHED: B separate matvecKernel launches, one per query.
    // -----------------------------------------------------------------
    int mvBlock = 128;
    int mvGrid = (d + mvBlock - 1) / mvBlock;

    auto launchUnbatched = [&]() {
        for (int q = 0; q < B; ++q) {
            matvecKernel<<<mvGrid, mvBlock>>>(W_d, X_d + static_cast<size_t>(q) * d, Yunb_d + static_cast<size_t>(q) * d, d);
        }
        CUDA_CHECK(cudaGetLastError());
    };

    // Warm-up (exclude one-time context/JIT costs from the timed runs).
    launchUnbatched();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Single-query latency: time one representative matvecKernel launch.
    GpuTimer singleTimer;
    singleTimer.start();
    matvecKernel<<<mvGrid, mvBlock>>>(W_d, X_d, Yunb_d, d);
    float singleQueryMs = singleTimer.stopAndGetMs();

    // Total latency to finish all B queries, unbatched (one at a time).
    GpuTimer unbatchedTimer;
    unbatchedTimer.start();
    launchUnbatched();
    CUDA_CHECK(cudaDeviceSynchronize());
    float unbatchedTotalMs = unbatchedTimer.stopAndGetMs();

    // -----------------------------------------------------------------
    // BATCHED: one tiled matmul launch computing all B outputs at once.
    // -----------------------------------------------------------------
    dim3 batBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 batGrid((d + TILE_WIDTH - 1) / TILE_WIDTH, (B + TILE_WIDTH - 1) / TILE_WIDTH);

    // Warm-up.
    batchedProjectionKernel<<<batGrid, batBlock>>>(X_d, W_d, Ybat_d, B, d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer batchedTimer;
    batchedTimer.start();
    batchedProjectionKernel<<<batGrid, batBlock>>>(X_d, W_d, Ybat_d, B, d);
    float batchedTotalMs = batchedTimer.stopAndGetMs();
    CUDA_CHECK(cudaGetLastError());

    std::vector<float> Yunb(static_cast<size_t>(B) * d), Ybat(static_cast<size_t>(B) * d);
    CUDA_CHECK(cudaMemcpy(Yunb.data(), Yunb_d, xBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(Ybat.data(), Ybat_d, xBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(W_d));
    CUDA_CHECK(cudaFree(X_d));
    CUDA_CHECK(cudaFree(Yunb_d));
    CUDA_CHECK(cudaFree(Ybat_d));

    float maxDiffUnb = 0.0f, maxDiffBat = 0.0f, maxDiffCross = 0.0f;
    bool okUnb = checkClose(Yunb, Yref, 1e-3f, &maxDiffUnb);
    bool okBat = checkClose(Ybat, Yref, 1e-3f, &maxDiffBat);
    bool okCross = checkClose(Yunb, Ybat, 1e-3f, &maxDiffCross);

    printf("Unbatched (per-query matvec) vs CPU reference: max|diff|=%.6f [%s]\n", maxDiffUnb, okUnb ? "match" : "MISMATCH");
    printf("Batched   (tiled matmul)     vs CPU reference: max|diff|=%.6f [%s]\n", maxDiffBat, okBat ? "match" : "MISMATCH");
    printf("Unbatched vs batched (cross-check): max|diff|=%.6f [%s]\n", maxDiffCross, okCross ? "match" : "MISMATCH");

    double unbThroughput = B / (unbatchedTotalMs / 1000.0);
    double batThroughput = B / (batchedTotalMs / 1000.0);
    printf("\n--- Sec.22.5 latency vs. throughput ---\n");
    printf("Single-query latency (unbatched, one launch):     %8.4f ms\n", singleQueryMs);
    printf("Total latency, unbatched, all %d queries:        %8.4f ms  (throughput = %.1f queries/s)\n", B, unbatchedTotalMs, unbThroughput);
    printf("Total/per-query latency, batched, all %d queries: %8.4f ms  (throughput = %.1f queries/s)\n", B, batchedTotalMs, batThroughput);
    printf("Throughput gain from batching: %.2fx\n", batThroughput / unbThroughput);
    printf("(Book: individual queries see longer latency once batched -- %.4f ms vs %.4f ms for a\n", batchedTotalMs, singleQueryMs);
    printf(" lone query -- but the whole group finishes sooner and throughput is higher.)\n");

    bool ok = okUnb && okBat && okCross;
    printf("\n%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
