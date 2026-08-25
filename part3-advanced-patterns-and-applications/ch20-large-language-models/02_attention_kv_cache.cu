// Chapter 20: Large language models
// §20.4, Figs. 20.5-20.7: incremental (autoregressive) attention with a
// growing KV cache.
//
// §20.4 observes that from one decoding iteration to the next, X gains
// exactly one new row (the newest token's embedding), and shows that this
// means Q, K, and V each only need one new row computed via a vector-matrix
// multiplication (Fig. 20.6: Q', K', V'); that all elements of QK^T with
// both indices < N are unchanged from the previous iteration; that the new
// N-th row of QK^T is a vector-matrix product between the new Q' row and
// the (cached) K^T, while the new N-th column is entirely masked to zero by
// the causality policy except for its diagonal element ("the new N th
// column of QK^T consists of all zero elements because of the zero-masking
// ... the only exception is the N th element of the N th column"); and that
// consequently all rows of O below the new one are unchanged too -- "one
// simply needs to perform a vector-matrix multiplication between the new
// row of softmax(QK^T) and the new V". §20.4: "this requirement can be met
// by memoizing K and V ... in the form of the so-called KV cache."
//
// This file simulates that mechanism directly on synthetic per-token Q, K,
// V rows (the vector-matrix projections against W_Q/W_K/W_V that produce
// them are the trivial, un-discussed part of §20.4's Fig. 20.6 -- the
// chapter's actual subject is what happens to K, V, and O once those rows
// exist). It reproduces the two phases of Fig. 20.7:
//
//   - prefill (summarization): the first N0 tokens' K, V rows fill the KV
//     cache directly (§20.4: "During the summarization phase, all
//     transformer layers generate their initial K and V matrices and fill
//     their KV caches");
//   - generation (decoding): for each subsequent token i = N0..N_SEQ-1, its
//     new Q_i/K_i/V_i row is appended to the cache (growing it to i+1
//     rows), and only the new row of S, P, and O is computed -- a GEMV
//     against the *entire* cached K^T/V, not a full N x N recompute. Since
//     the new row can only attend to columns <= i (all of which now exist
//     in the cache) and there is no later column yet, causality holds
//     automatically with no explicit masking needed, exactly as §20.4
//     describes for the new row.
//
// At every generation step, the incrementally produced O_i row is checked
// against a full causal-softmax recompute over the first (i+1) tokens
// (mirroring file 01's naive CPU reference, computed here in double
// precision) -- i.e. this file verifies the KV-cache shortcut is
// numerically equivalent to full recomputation at every step, per the task.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cub/cub.cuh>

#include "../../common/cuda_utils.h"

#define N_SEQ 64          // final sequence length N
#define N_PREFILL 8        // §20.4 prefill (summarization) phase length
#define D_HEAD 32          // head dimension d
#define BLOCK_SIZE 128      // softmax/GEMV reduction threads per step

// Looser-than-default epsilon for the same reason as file 01: softmax's
// exp/sum/normalize chain accumulates rounding differently on the GPU
// (parallel tree reduction) than the CPU double-precision reference.
static const float kAttnEps = 2e-3f;

// ---------------------------------------------------------------------------
// New row of S: S_i[j] = scale * dot(Q_i, K_cache[j]), j = 0..rowLen-1.
// One block, one thread per cached column (BLOCK_SIZE-strided if rowLen
// exceeds BLOCK_SIZE). No causal masking needed: the new row's valid range
// is exactly [0, rowLen), which is exactly what the KV cache holds so far
// (§20.4: the new row is never masked within its own valid range).
// ---------------------------------------------------------------------------
__global__ void newRowSKernel(const float *Q_new, const float *K_cache, float *S_new, int rowLen, float scale) {
    for (int j = threadIdx.x; j < rowLen; j += BLOCK_SIZE) {
        float acc = 0.0f;
        for (int k = 0; k < D_HEAD; ++k) acc += Q_new[k] * K_cache[j * D_HEAD + k];
        S_new[j] = acc * scale;
    }
}

// Softmax of the new row (length rowLen, all valid -- no masking), same
// max-then-sum block reduction structure as file 01's Fig. 20.4 kernel.
__global__ void newRowSoftmaxKernel(const float *S_new, float *P_new, int rowLen) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float max_or_sum;

    float max_val_thread = -INFINITY;
    for (int j = threadIdx.x; j < rowLen; j += BLOCK_SIZE) max_val_thread = fmaxf(max_val_thread, S_new[j]);
    __syncthreads();
    float max_val = BlockReduce(temp_storage).Reduce(max_val_thread, cub::Max());
    if (threadIdx.x == 0) max_or_sum = max_val;
    __syncthreads();
    max_val = max_or_sum;

    float sum_thread = 0.0f;
    for (int j = threadIdx.x; j < rowLen; j += BLOCK_SIZE) sum_thread += expf(S_new[j] - max_val);
    __syncthreads();
    float sum = BlockReduce(temp_storage).Sum(sum_thread);
    if (threadIdx.x == 0) max_or_sum = sum;
    __syncthreads();
    sum = max_or_sum;

    for (int j = threadIdx.x; j < rowLen; j += BLOCK_SIZE) P_new[j] = expf(S_new[j] - max_val) / sum;
}

// New row of O: O_i[c] = sum_j P_i[j] * V_cache[j][c] -- a GEMV against the
// entire cached V, per §20.4's "vector-matrix multiplication between the
// new row of softmax(QK^T) and the new V".
__global__ void newRowOKernel(const float *P_new, const float *V_cache, float *O_new, int rowLen) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= D_HEAD) return;
    float acc = 0.0f;
    for (int j = 0; j < rowLen; ++j) acc += P_new[j] * V_cache[j * D_HEAD + c];
    O_new[c] = acc;
}

// ---------------------------------------------------------------------------
// CPU reference: full causal-softmax attention (double precision), for a
// prefix of length `len`, evaluated only at the last row (index len-1) --
// this is file 01's naive formulation, mirrored locally per this repo's
// no-cross-file-includes convention, used as the "full recomputation"
// ground truth that the KV-cache shortcut is checked against at each step.
// ---------------------------------------------------------------------------
static void cpuNaiveAttentionLastRow(const std::vector<float> &Q, const std::vector<float> &K,
                                      const std::vector<float> &V, int len, std::vector<float> &O_last) {
    double scale = 1.0 / std::sqrt(static_cast<double>(D_HEAD));
    int r = len - 1;
    std::vector<double> row(len);
    double max_val = -INFINITY;
    for (int c = 0; c < len; ++c) {
        double acc = 0.0;
        for (int k = 0; k < D_HEAD; ++k)
            acc += static_cast<double>(Q[r * D_HEAD + k]) * static_cast<double>(K[c * D_HEAD + k]);
        row[c] = acc * scale;
        if (row[c] > max_val) max_val = row[c];
    }
    double sum = 0.0;
    for (int c = 0; c < len; ++c) {
        row[c] = std::exp(row[c] - max_val);
        sum += row[c];
    }
    for (int c = 0; c < len; ++c) row[c] /= sum;
    O_last.assign(D_HEAD, 0.0f);
    for (int d = 0; d < D_HEAD; ++d) {
        double acc = 0.0;
        for (int c = 0; c < len; ++c) acc += row[c] * static_cast<double>(V[c * D_HEAD + d]);
        O_last[d] = static_cast<float>(acc);
    }
}

static std::vector<float> randomVec(size_t n, unsigned int seed) {
    std::vector<float> v(n);
    unsigned int state = seed;
    auto nextRand = [&]() -> float {
        state = state * 1103515245u + 12345u;
        return static_cast<float>((state >> 8) & 0xFFFFFFu) / static_cast<float>(0xFFFFFFu);
    };
    for (size_t i = 0; i < n; ++i) v[i] = 2.0f * nextRand() - 1.0f;
    return v;
}

int main() {
    printf("Incremental attention with a growing KV cache (§20.4, Figs. 20.5-20.7):\n");
    printf("N=%d, prefill=%d, d=%d, softmax eps=%.4f\n", N_SEQ, N_PREFILL, D_HEAD, kAttnEps);

    // The full Q/K/V "ground truth" for all N_SEQ tokens -- per-token rows
    // are revealed one at a time during the generation phase, exactly as
    // they would be produced by the (unmodeled) W_Q/W_K/W_V projections.
    std::vector<float> Q = randomVec(static_cast<size_t>(N_SEQ) * D_HEAD, 11u);
    std::vector<float> K = randomVec(static_cast<size_t>(N_SEQ) * D_HEAD, 22u);
    std::vector<float> V = randomVec(static_cast<size_t>(N_SEQ) * D_HEAD, 33u);

    size_t qkvBytes = static_cast<size_t>(N_SEQ) * D_HEAD * sizeof(float);
    float *K_cache_d, *V_cache_d, *S_new_d, *P_new_d, *O_new_d;
    CUDA_CHECK(cudaMalloc(&K_cache_d, qkvBytes));
    CUDA_CHECK(cudaMalloc(&V_cache_d, qkvBytes));
    CUDA_CHECK(cudaMalloc(&S_new_d, N_SEQ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&P_new_d, N_SEQ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&O_new_d, D_HEAD * sizeof(float)));

    // Prefill (summarization) phase: fill the KV cache directly with the
    // first N_PREFILL tokens' K, V rows (§20.4: "all transformer layers
    // generate their initial K and V matrices and fill their KV caches").
    size_t prefillBytes = static_cast<size_t>(N_PREFILL) * D_HEAD * sizeof(float);
    CUDA_CHECK(cudaMemcpy(K_cache_d, K.data(), prefillBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(V_cache_d, V.data(), prefillBytes, cudaMemcpyHostToDevice));

    float scale = 1.0f / sqrtf(static_cast<float>(D_HEAD));
    dim3 oBlock(32);
    dim3 oGrid((D_HEAD + oBlock.x - 1) / oBlock.x);

    bool ok = true;
    float maxDiffOverall = 0.0f;
    GpuTimer timer;
    float totalMs = 0.0f;

    // Generation (decoding) phase: one new token at a time, growing the KV
    // cache row by row (Fig. 20.6/20.7).
    for (int i = N_PREFILL; i < N_SEQ; ++i) {
        int rowLen = i + 1;  // causal prefix length for the new row, incl. itself

        // Append this step's new K_i, V_i row to the cache (§20.4: "The LLM
        // attaches these new rows to the K and V matrices from the
        // previous iteration in the KV cache").
        CUDA_CHECK(cudaMemcpy(K_cache_d + static_cast<size_t>(i) * D_HEAD, &K[static_cast<size_t>(i) * D_HEAD],
                               D_HEAD * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(V_cache_d + static_cast<size_t>(i) * D_HEAD, &V[static_cast<size_t>(i) * D_HEAD],
                               D_HEAD * sizeof(float), cudaMemcpyHostToDevice));

        const float *Q_new_h = &Q[static_cast<size_t>(i) * D_HEAD];
        float *Q_new_d;
        CUDA_CHECK(cudaMalloc(&Q_new_d, D_HEAD * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(Q_new_d, Q_new_h, D_HEAD * sizeof(float), cudaMemcpyHostToDevice));

        timer.start();
        newRowSKernel<<<1, BLOCK_SIZE>>>(Q_new_d, K_cache_d, S_new_d, rowLen, scale);
        CUDA_CHECK(cudaGetLastError());
        newRowSoftmaxKernel<<<1, BLOCK_SIZE>>>(S_new_d, P_new_d, rowLen);
        CUDA_CHECK(cudaGetLastError());
        newRowOKernel<<<oGrid, oBlock>>>(P_new_d, V_cache_d, O_new_d, rowLen);
        CUDA_CHECK(cudaGetLastError());
        totalMs += timer.stopAndGetMs();

        std::vector<float> O_new(D_HEAD);
        CUDA_CHECK(cudaMemcpy(O_new.data(), O_new_d, D_HEAD * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(Q_new_d));

        std::vector<float> O_ref;
        cpuNaiveAttentionLastRow(Q, K, V, rowLen, O_ref);

        for (int d = 0; d < D_HEAD; ++d) {
            float diff = fabsf(O_new[d] - O_ref[d]);
            if (diff > maxDiffOverall) maxDiffOverall = diff;
            if (!nearlyEqual(O_new[d], O_ref[d], kAttnEps)) {
                ok = false;
                fprintf(stderr, "step i=%d: mismatch at d=%d: gpu=%.6f ref=%.6f\n", i, d, O_new[d], O_ref[d]);
            }
        }
    }

    printf("generation steps %d..%d: max|diff|=%.6f, total kernel time=%.4f ms  [%s]\n", N_PREFILL, N_SEQ - 1,
           maxDiffOverall, totalMs, ok ? "match" : "MISMATCH");

    CUDA_CHECK(cudaFree(K_cache_d));
    CUDA_CHECK(cudaFree(V_cache_d));
    CUDA_CHECK(cudaFree(S_new_d));
    CUDA_CHECK(cudaFree(P_new_d));
    CUDA_CHECK(cudaFree(O_new_d));

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
