// Chapter 16: Dynamic programming and wavefront parallelism
// §16.5-16.6: basic (untiled) parallel Smith-Waterman local sequence
// alignment -- one kernel launch per anti-diagonal wavefront, one GPU
// thread per anti-diagonal cell.
//
// §16.5 defines the scoring (homology) matrix H over two sequences seqR
// ("read", rows) and seqC ("reference", columns), both length L_seq, so H is
// an L x L matrix with L = L_seq + 1 (row 0 / column 0 are boundary zeros).
// Eq. (16.2):
//   H[r][q] = max(H[r-1][q-1] + S(r,q), H[r-1][q] - insertion_penalty,
//                 H[r][q-1] - deletion_penalty, 0)
// where S(r,q) is +MATCH if seqR[r-1] == seqC[q-1], else MISMATCH.
// §16.5 observes that H[r][q] depends only on its three neighbor cells in
// the two previous anti-diagonals (Fig. 16.6), so every cell of one
// anti-diagonal (r + q constant) is independent and can be computed in
// parallel -- each anti-diagonal is a wavefront, and (unlike Floyd-Warshall)
// this wavefront's size grows then shrinks (Fig. 16.2(c)).
//
// §16.6 opens by describing exactly this basic parallelization -- "launches
// one kernel for each anti-diagonal wavefront, and each GPU thread computes
// one anti-diagonal cell" -- as the baseline that the section's block-tiling
// technique (implemented separately in 03_wavefront_block_tiled.cu)
// improves upon. The book states "We leave the implementation of this basic
// parallelization as an exercise" without a code figure, but the mechanism
// itself (one kernel launch per diagonal, one thread per cell, using Eq.
// 16.2) is fully specified by the surrounding text, so it is implemented
// here as file 02, ahead of the tiled version in file 03.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <random>
#include <algorithm>

#include "../../common/cuda_utils.h"

#define MATCH 3
#define MISMATCH (-3)
#define INSERTION (-2)  // penalty for an insertion (applied to the "north" neighbor)
#define DELETION (-2)   // penalty for a deletion (applied to the "west" neighbor)
#define THREADS 128

// ---------------------------------------------------------------------------
// §16.5/§16.6: one thread computes one cell (r, q) of anti-diagonal d
// (r + q == d). rMin/rMax bound the valid r values on this diagonal; the
// host computes rMin once per launch (mirrors how many cells the wavefront
// currently has, per §16.3's "wavefront size may grow or shrink").
// ---------------------------------------------------------------------------
__global__ void swBasicKernel(int *H, const char *seqR, const char *seqC, int L, int d, int rMin, int rMax) {
    int r = rMin + blockIdx.x * blockDim.x + threadIdx.x;
    if (r > rMax) return;
    int q = d - r;

    int nw = H[(r - 1) * L + (q - 1)];
    int n = H[(r - 1) * L + q];
    int w = H[r * L + (q - 1)];
    int subs = (seqR[r - 1] == seqC[q - 1]) ? MATCH : MISMATCH;

    int best = 0;
    int diag = nw + subs;
    if (diag > best) best = diag;
    int north = n + INSERTION;
    if (north > best) best = north;
    int west = w + DELETION;
    if (west > best) best = west;

    H[r * L + q] = best;
}

// ---------------------------------------------------------------------------
// CPU reference: same recurrence (Eq. 16.2), evaluated in row-major order,
// which also happens to respect the (r-1,*) / (r,q-1) dependencies.
// ---------------------------------------------------------------------------
void smithWatermanCPU(std::vector<int> &H, const std::string &seqR, const std::string &seqC, int L_seq) {
    int L = L_seq + 1;
    for (int r = 1; r < L; ++r) {
        for (int q = 1; q < L; ++q) {
            int subs = (seqR[r - 1] == seqC[q - 1]) ? MATCH : MISMATCH;
            int best = 0;
            best = std::max(best, H[(r - 1) * L + (q - 1)] + subs);
            best = std::max(best, H[(r - 1) * L + q] + INSERTION);
            best = std::max(best, H[r * L + (q - 1)] + DELETION);
            H[r * L + q] = best;
        }
    }
}

std::string generateSequence(int len, unsigned int seed) {
    static const char alphabet[4] = {'A', 'C', 'G', 'T'};
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pick(0, 3);
    std::string s(len, 'A');
    for (int i = 0; i < len; ++i) s[i] = alphabet[pick(rng)];
    return s;
}

// ---------------------------------------------------------------------------
// Host driver: one kernel launch per anti-diagonal d = 2 .. 2*(L-1).
// ---------------------------------------------------------------------------
float runSmithWatermanBasicGPU(std::vector<int> &H_h, const std::string &seqR, const std::string &seqC, int L_seq) {
    int L = L_seq + 1;
    size_t hBytes = static_cast<size_t>(L) * L * sizeof(int);

    int *H_d;
    char *seqR_d, *seqC_d;
    CUDA_CHECK(cudaMalloc(&H_d, hBytes));
    CUDA_CHECK(cudaMalloc(&seqR_d, L_seq));
    CUDA_CHECK(cudaMalloc(&seqC_d, L_seq));
    CUDA_CHECK(cudaMemcpy(H_d, H_h.data(), hBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(seqR_d, seqR.data(), L_seq, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(seqC_d, seqC.data(), L_seq, cudaMemcpyHostToDevice));

    GpuTimer timer;
    timer.start();
    for (int d = 2; d <= 2 * (L - 1); ++d) {
        int rMin = std::max(1, d - (L - 1));
        int rMax = std::min(L - 1, d - 1);
        int count = rMax - rMin + 1;
        int blocks = (count + THREADS - 1) / THREADS;
        swBasicKernel<<<blocks, THREADS>>>(H_d, seqR_d, seqC_d, L, d, rMin, rMax);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms = timer.stopAndGetMs();

    CUDA_CHECK(cudaMemcpy(H_h.data(), H_d, hBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(H_d));
    CUDA_CHECK(cudaFree(seqR_d));
    CUDA_CHECK(cudaFree(seqC_d));
    return ms;
}

bool runTestCase(int L_seq) {
    std::string seqR = generateSequence(L_seq, 100u + static_cast<unsigned int>(L_seq));
    std::string seqC = generateSequence(L_seq, 200u + static_cast<unsigned int>(L_seq) * 7u);

    int L = L_seq + 1;
    std::vector<int> H_cpu(static_cast<size_t>(L) * L, 0);
    smithWatermanCPU(H_cpu, seqR, seqC, L_seq);

    std::vector<int> H_gpu(static_cast<size_t>(L) * L, 0);
    float ms = runSmithWatermanBasicGPU(H_gpu, seqR, seqC, L_seq);

    bool ok = (H_gpu == H_cpu);
    int bestCpu = *std::max_element(H_cpu.begin(), H_cpu.end());
    int bestGpu = *std::max_element(H_gpu.begin(), H_gpu.end());
    printf("L_seq=%-5d best_score(cpu)=%-4d best_score(gpu)=%-4d %.4f ms  [%s]\n", L_seq, bestCpu, bestGpu, ms,
           ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    printf("Basic wavefront Smith-Waterman: one kernel per anti-diagonal (§16.5-16.6):\n");
    bool ok = true;
    ok = runTestCase(1) && ok;
    ok = runTestCase(2) && ok;
    ok = runTestCase(63) && ok;   // not a multiple of THREADS
    ok = runTestCase(256) && ok;
    ok = runTestCase(513) && ok;  // odd, exercises growing/shrinking diagonal bounds

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
