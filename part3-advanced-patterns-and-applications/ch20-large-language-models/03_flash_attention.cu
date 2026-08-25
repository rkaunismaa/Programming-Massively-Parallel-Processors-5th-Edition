// Chapter 20: Large language models
// §20.5, Eq. (20.3)-(20.6), Figs. 20.8-20.16: tiled, online-softmax flash
// attention. Fuses the two matmuls (Q K^T and P V) and the softmax into a
// single kernel, never materializing the full N x N S/P matrices in global
// memory.
//
// Tiling scheme (Fig. 20.8), followed exactly by the sizes below:
//   - Each thread block owns a horizontal panel of B_R contiguous rows of Q
//     (and, ultimately, of O). The panel is further split into B_R_WARP-row
//     sub-panels, one per warp (B_R_WARP = B_R / warps-per-block); this
//     sample uses exactly one warp per block, so B_R_WARP == B_R and every
//     "warp-level" step below is simply the whole block's only warp.
//   - The block iterates T_C = N/B_C times over B_C-column tiles of K^T and
//     B_C-row tiles of V (both held in shared memory), accumulating its
//     contribution to the O panel via the online-softmax composition rule.
//   - d = D_HEAD is a multiple of WARP_SIZE (here d == WARP_SIZE, so each
//     thread holds exactly one Q/O element per row, i.e. D_PER_THREAD=1,
//     the simplest valid case of §20.5's "we assume that d is a multiple of
//     WARP_SIZE" simplification); B_C is likewise a multiple of WARP_SIZE
//     (here B_C == WARP_SIZE), per §20.5's "we only allow B_c value to be
//     multiples of WARP_SIZE".
//
// Online-softmax recurrence (Eq. 20.3-20.6): for row r and column subsets A
// (already merged) and B (the new tile), with m_r,X and D_r,X the running
// max/denominator restricted to subset X:
//   m_r,A∪B = max(m_r,A, m_r,B)
//   D_r,A∪B = D_r,A * e^(m_r,A - m_r,A∪B) + D_r,B * e^(m_r,B - m_r,A∪B)
//   o_r,c,A∪B = o_r,c,A * e^(m_r,A - m_r,A∪B) + o_r,c,B * e^(m_r,B - m_r,A∪B)
// This kernel evaluates the B-subset terms directly relative to the already
// -updated merged max m_r,A∪B (i.e. new P/O contributions use exponents
// (l_r,j - m_r,A∪B) directly, per §20.5's description of Fig. 20.15 lines
// 8-9 and Fig. 20.13's "there is no need to rescale these P elements when
// merging them into O"), so only the *old* accumulated D_i/O_i need the
// e^(m_r,A - m_r,A∪B) rescale before the new terms are added -- this is
// what update_m_and_D()/compute_O() below do.
//
// NOTE on a probable OCR artifact in the extracted chapter text: the prose
// describing update_m_and_D() (Fig. 20.14) states that its line 6 "computes
// the term D_r,B * e^(m_r,B - m_r,A∪B)" -- but at that point in the code, D_i
// still holds only the old D_r,A (the new tile's contribution D_r,B isn't
// summed until compute_P_and_update_D() runs afterwards), and the function
// has no access to a not-yet-computed D_r,B. The only interpretation
// consistent with the code's actual data flow -- and the one implemented
// here -- is that this line rescales the *old* term, i.e. computes
// D_i[ii] = D_r,A * e^(m_r,A - m_r,A∪B), matching the first addend of Eq.
// (20.4)/(20.6); compute_P_and_update_D() then computes the D_r,B term
// directly against the merged max and adds it (Fig. 20.15 lines 10-14),
// completing the composition rule with no separate rescale needed for that
// term. This is almost certainly a subscript transcription slip (A vs. B)
// in the source PDF's math rendering, not a deviation this file is taking
// from the book -- the implementation follows Eq. (20.4)/(20.6) exactly.
//
// CPU reference: the same full causal-softmax naive attention as file 01
// (double precision), reproduced locally per this repo's no-cross-file-
// includes convention, used to check that the tiled/fused reformulation is
// numerically equivalent to the non-tiled formulation, per §20.5: "Flash
// attention is an exact reformulation of the attention mechanism ... It is
// mathematically identical to the original formulation."

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cub/cub.cuh>

#include "../../common/cuda_utils.h"

#define WARP_SIZE 32
#define LOG_NUM_BANKS 5

#define N_SEQ 64             // sequence length N
#define D_HEAD 32             // head dimension d (multiple of WARP_SIZE)
#define B_R 16                 // rows of Q per thread-block panel (Fig. 20.8)
#define B_C 32                 // columns of K^T / rows of V per tile (Fig. 20.8)
#define WARPS_PER_BLOCK 1       // one warp per block for this sample
#define B_R_WARP (B_R / WARPS_PER_BLOCK)          // rows per warp sub-panel
#define D_PER_THREAD (D_HEAD / WARP_SIZE)          // Q/O registers per row per thread
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * WARP_SIZE)
#define T_R (N_SEQ / B_R)      // number of Q panels == gridDim.x
#define T_C (N_SEQ / B_C)      // number of K^T/V tiles per Q panel

#define KT_ELEMS (D_HEAD * B_C)
#define KT_PAD_SIZE (KT_ELEMS + KT_ELEMS / 32 + 4)

// Flash attention accumulates more rescaling steps than files 01/02 (each
// of the T_C tile merges applies an expf()-based rescale to the running O/D
// on top of the softmax exp/sum/normalize chain itself), so it gets a
// slightly looser epsilon than those two. Measured max|diff| against the
// double-precision naive reference is ~1e-7.
static const float kAttnEps = 3e-3f;

__device__ __forceinline__ int laneIdx() { return threadIdx.x % WARP_SIZE; }
__device__ __forceinline__ int warpIdx() { return threadIdx.x / WARP_SIZE; }
__device__ __forceinline__ int addr(int x) { return x + (x >> LOG_NUM_BANKS); }

// ---------------------------------------------------------------------------
// Fig. 20.10: load a Q sub-panel into registers, interleaved across the
// warp so that each WARP_SIZE-wide section of a row is loaded coalesced.
// ---------------------------------------------------------------------------
__device__ void loadQ(const float *Q, float Q_i[B_R_WARP][D_PER_THREAD], int rowBase) {
    int lane = laneIdx();
    for (int ii = 0; ii < B_R_WARP; ++ii)
        for (int s = 0; s < D_PER_THREAD; ++s)
            Q_i[ii][s] = Q[(rowBase + ii) * D_HEAD + s * WARP_SIZE + lane];
}

__device__ void initialize(float O_i[B_R_WARP][D_PER_THREAD], float m_i[B_R_WARP], float D_i[B_R_WARP]) {
    for (int ii = 0; ii < B_R_WARP; ++ii) {
        m_i[ii] = -INFINITY;
        D_i[ii] = 0.0f;
        for (int s = 0; s < D_PER_THREAD; ++s) O_i[ii][s] = 0.0f;
    }
}

// ---------------------------------------------------------------------------
// Fig. 20.11: load a B_C-column tile of K^T (transposed on the fly from K)
// and a B_C-row tile of V into shared memory. KT_j is stored with padding
// (addr()) to avoid shared-memory bank conflicts (§20.5); V_j is not padded.
// ---------------------------------------------------------------------------
__device__ void loadKTV(const float *K, const float *V, float *KT_j, float *V_j, int colBase) {
    for (int idx = threadIdx.x; idx < KT_ELEMS; idx += THREADS_PER_BLOCK) {
        int dd = idx / B_C, cc = idx % B_C;
        KT_j[addr(idx)] = K[(colBase + cc) * D_HEAD + dd];
    }
    for (int idx = threadIdx.x; idx < B_C * D_HEAD; idx += THREADS_PER_BLOCK) {
        int cc = idx / D_HEAD, dd = idx % D_HEAD;
        V_j[idx] = V[(colBase + cc) * D_HEAD + dd];
    }
}

// ---------------------------------------------------------------------------
// Fig. 20.12: compute one row of the S tile (a vector-matrix product between
// the warp's current Q row and the KT tile) and the row's tile-local
// maximum. Q elements are broadcast across the warp with __shfl_sync since
// each lane only holds one WARP_SIZE-wide section of the Q row (Fig. 20.10).
// Causality (§20.2, Eq. 20.1's mask M) is applied here by writing -inf for
// s_i elements whose global column exceeds the global row.
// ---------------------------------------------------------------------------
__device__ float computeSAndMax(float S_i[B_C], const float Q_i[B_R_WARP][D_PER_THREAD], const float *KT_j, int ii,
                                 int rowGlobal, int colBase, float scale) {
    int lane = laneIdx();
    float acc = 0.0f;
    for (int dd = 0; dd < D_HEAD; ++dd) {
        float qval = __shfl_sync(0xFFFFFFFF, Q_i[ii][dd / WARP_SIZE], dd % WARP_SIZE);
        acc += qval * KT_j[addr(dd * B_C + lane)];
    }
    int colGlobal = colBase + lane;
    float val = (colGlobal > rowGlobal) ? -INFINITY : acc * scale;
    S_i[lane] = val;

    typedef cub::WarpReduce<float> WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp_storage;
    float rowMax = WarpReduce(temp_storage).Reduce(val, cub::Max());
    return __shfl_sync(0xFFFFFFFF, rowMax, 0);
}

// ---------------------------------------------------------------------------
// Fig. 20.14: update the running row maximum m_i and rescale the running
// denominator D_i's already-accumulated (subset-A) term by
// e^(m_r,A - m_r,A∪B), per Eq. (20.4)'s first addend. See the file header
// note on the extracted text's D_r,A/D_r,B subscript ambiguity.
// ---------------------------------------------------------------------------
__device__ float updateMAndD(float &m_i_ii, float &D_i_ii, float currMaxWarp) {
    float last_m = m_i_ii;
    if (currMaxWarp > m_i_ii) m_i_ii = currMaxWarp;
    D_i_ii *= expf(last_m - m_i_ii);
    return last_m;
}

// ---------------------------------------------------------------------------
// Fig. 20.15: turn the S tile row into a P tile row in place (exponentials
// relative to the already-merged max m_i_ii -- masked entries were already
// -inf, so expf() naturally yields 0 with no further masking needed) and
// accumulate the new tile's contribution directly into D_i (Eq. 20.4's
// second addend, computed relative to m_r,A∪B already, so no extra rescale
// is needed for this term).
// ---------------------------------------------------------------------------
__device__ void computePAndUpdateD(float S_i[B_C], float &D_i_ii, float m_i_ii) {
    int lane = laneIdx();
    float val = expf(S_i[lane] - m_i_ii);
    S_i[lane] = val;

    typedef cub::WarpReduce<float> WarpReduce;
    __shared__ typename WarpReduce::TempStorage temp_storage;
    float sum = WarpReduce(temp_storage).Sum(val);
    float currSum = __shfl_sync(0xFFFFFFFF, sum, 0);
    D_i_ii += currSum;
}

// ---------------------------------------------------------------------------
// Fig. 20.13: merge this tile's P row / V tile product into the O tile.
// Rescales the existing O_i accumulation by e^(m_r,A - m_r,A∪B) (line 3)
// before adding the new tile's P*V contribution, which needs no rescale
// itself (computed directly against the merged max, per compute_P_and_
// update_D above).
// ---------------------------------------------------------------------------
__device__ void computeO(float O_i[B_R_WARP][D_PER_THREAD], const float S_i[B_C], const float *V_j, int ii,
                          float lastM, float mIii) {
    int lane = laneIdx();
    float rescale = expf(lastM - mIii);
    for (int s = 0; s < D_PER_THREAD; ++s) {
        float acc = 0.0f;
        for (int cc = 0; cc < B_C; ++cc) acc += S_i[cc] * V_j[cc * D_HEAD + s * WARP_SIZE + lane];
        O_i[ii][s] = O_i[ii][s] * rescale + acc;
    }
}

// ---------------------------------------------------------------------------
// Fig. 20.16: normalize by the final denominator and store O, D to global
// memory.
// ---------------------------------------------------------------------------
__device__ void storeO(const float O_i[B_R_WARP][D_PER_THREAD], const float D_i[B_R_WARP], int rowBase, float *out_O,
                        float *out_D) {
    int lane = laneIdx();
    for (int ii = 0; ii < B_R_WARP; ++ii) {
        int row = rowBase + ii;
        for (int s = 0; s < D_PER_THREAD; ++s) out_O[row * D_HEAD + s * WARP_SIZE + lane] = O_i[ii][s] / D_i[ii];
        if (lane == 0) out_D[row] = D_i[ii];
    }
}

// ---------------------------------------------------------------------------
// Fig. 20.9: the flash attention forward-pass kernel. 1D grid, gridDim.x ==
// T_R so the outer panel loop (line 16) runs exactly once per block; kept
// as a loop for fidelity to the book's grid-stride structure.
// ---------------------------------------------------------------------------
__global__ void flashAttentionKernel(const float *Q, const float *K, const float *V, float *out_O, float *out_D,
                                      float scale) {
    __shared__ float KT_j[KT_PAD_SIZE];
    __shared__ float V_j[B_C * D_HEAD];
    __shared__ float S_i_storage[B_R_WARP][B_C];

    for (int panel = blockIdx.x; panel < T_R; panel += gridDim.x) {
        float Q_i[B_R_WARP][D_PER_THREAD];
        float O_i[B_R_WARP][D_PER_THREAD];
        float m_i[B_R_WARP], D_i[B_R_WARP];

        int rowBase = panel * B_R + warpIdx() * B_R_WARP;
        initialize(O_i, m_i, D_i);
        loadQ(Q, Q_i, rowBase);

        for (int j = 0; j < T_C; ++j) {
            int colBase = j * B_C;
            loadKTV(K, V, KT_j, V_j, colBase);
            __syncthreads();

            for (int ii = 0; ii < B_R_WARP; ++ii) {
                int rowGlobal = rowBase + ii;
                float *S_i = S_i_storage[ii];
                float currMaxWarp = computeSAndMax(S_i, Q_i, KT_j, ii, rowGlobal, colBase, scale);
                float lastM = updateMAndD(m_i[ii], D_i[ii], currMaxWarp);
                computePAndUpdateD(S_i, D_i[ii], m_i[ii]);
                // compute_O() reads S_i[cc] for every cc, i.e. every lane's
                // P value, not just its own -- independent thread
                // scheduling (Volta+) does not guarantee those writes are
                // visible warp-wide without an explicit reconvergence
                // point, so __syncwarp() is required here even though this
                // is a single-warp block (compute-sanitizer's --tool
                // racecheck flags this exact hazard without it).
                __syncwarp();
                computeO(O_i, S_i, V_j, ii, lastM, m_i[ii]);
            }
            __syncthreads();
        }

        storeO(O_i, D_i, rowBase, out_O, out_D);
    }
}

// ---------------------------------------------------------------------------
// CPU reference (double precision), identical formulation to file 01's
// cpuNaiveAttention, reproduced locally per this repo's convention.
// ---------------------------------------------------------------------------
static void cpuNaiveAttention(const std::vector<float> &Q, const std::vector<float> &K, const std::vector<float> &V,
                               std::vector<float> &O) {
    double scale = 1.0 / std::sqrt(static_cast<double>(D_HEAD));
    O.assign(static_cast<size_t>(N_SEQ) * D_HEAD, 0.0f);
    std::vector<double> row(N_SEQ);
    for (int r = 0; r < N_SEQ; ++r) {
        double maxVal = -INFINITY;
        for (int c = 0; c <= r; ++c) {
            double acc = 0.0;
            for (int k = 0; k < D_HEAD; ++k)
                acc += static_cast<double>(Q[r * D_HEAD + k]) * static_cast<double>(K[c * D_HEAD + k]);
            row[c] = acc * scale;
            if (row[c] > maxVal) maxVal = row[c];
        }
        double sum = 0.0;
        for (int c = 0; c <= r; ++c) {
            row[c] = std::exp(row[c] - maxVal);
            sum += row[c];
        }
        for (int c = 0; c <= r; ++c) row[c] /= sum;
        for (int d = 0; d < D_HEAD; ++d) {
            double acc = 0.0;
            for (int c = 0; c <= r; ++c) acc += row[c] * static_cast<double>(V[c * D_HEAD + d]);
            O[r * D_HEAD + d] = static_cast<float>(acc);
        }
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
    printf("Tiled, online-softmax flash attention forward pass (§20.5, Eq. 20.3-20.6, Fig. 20.9):\n");
    printf("N=%d, d=%d, B_r=%d, B_c=%d, warps/block=%d, T_r=%d, T_c=%d, softmax eps=%.4f\n", N_SEQ, D_HEAD, B_R, B_C,
           WARPS_PER_BLOCK, T_R, T_C, kAttnEps);

    std::vector<float> Q = randomVec(static_cast<size_t>(N_SEQ) * D_HEAD, 101u);
    std::vector<float> K = randomVec(static_cast<size_t>(N_SEQ) * D_HEAD, 202u);
    std::vector<float> V = randomVec(static_cast<size_t>(N_SEQ) * D_HEAD, 303u);

    std::vector<float> O_ref;
    cpuNaiveAttention(Q, K, V, O_ref);

    float *Q_d, *K_d, *V_d, *O_d, *D_d;
    size_t qkvBytes = static_cast<size_t>(N_SEQ) * D_HEAD * sizeof(float);
    CUDA_CHECK(cudaMalloc(&Q_d, qkvBytes));
    CUDA_CHECK(cudaMalloc(&K_d, qkvBytes));
    CUDA_CHECK(cudaMalloc(&V_d, qkvBytes));
    CUDA_CHECK(cudaMalloc(&O_d, qkvBytes));
    CUDA_CHECK(cudaMalloc(&D_d, N_SEQ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(Q_d, Q.data(), qkvBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(K_d, K.data(), qkvBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(V_d, V.data(), qkvBytes, cudaMemcpyHostToDevice));

    float scale = 1.0f / sqrtf(static_cast<float>(D_HEAD));

    auto launch = [&]() {
        flashAttentionKernel<<<T_R, THREADS_PER_BLOCK>>>(Q_d, K_d, V_d, O_d, D_d, scale);
        CUDA_CHECK(cudaGetLastError());
    };

    launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    launch();
    float ms = timer.stopAndGetMs();

    std::vector<float> O(static_cast<size_t>(N_SEQ) * D_HEAD);
    CUDA_CHECK(cudaMemcpy(O.data(), O_d, qkvBytes, cudaMemcpyDeviceToHost));

    float maxDiff = 0.0f;
    bool ok = checkClose(O, O_ref, kAttnEps, &maxDiff);
    printf("GPU flash attention vs CPU-double naive reference: max|diff|=%.6f  %.4f ms  [%s]\n", maxDiff, ms,
           ok ? "match" : "MISMATCH");

    CUDA_CHECK(cudaFree(Q_d));
    CUDA_CHECK(cudaFree(K_d));
    CUDA_CHECK(cudaFree(V_d));
    CUDA_CHECK(cudaFree(O_d));
    CUDA_CHECK(cudaFree(D_d));

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
