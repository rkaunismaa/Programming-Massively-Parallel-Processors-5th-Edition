// Chapter 16: Dynamic programming and wavefront parallelism
// §16.6, Figs. 16.7-16.12: thread block-level tiling applied to
// Smith-Waterman's wavefront parallelization.
//
// §16.6 opens: "A basic GPU parallelization of the Smith-Waterman algorithm
// launches one kernel for each anti-diagonal wavefront..." and the whole
// section (host code Fig. 16.8, kernel Fig. 16.9, device helpers Figs.
// 16.10-16.12) is built exclusively around Smith-Waterman's scoring matrix
// -- Floyd-Warshall (§16.4) is never revisited here. So this file
// reimplements Smith-Waterman's block-tiled kernel, not Floyd-Warshall's.
//
// The dynamic programming matrix is divided into `threads x threads` tiles,
// one thread block per tile (Fig. 16.7(b)). Inside a tile, threads compute
// anti-diagonal entries of the tile and synchronize locally with
// __syncthreads() (Fig. 16.7(c)); global synchronization (kernel
// termination + relaunch) only happens between "tile anti-diagonals" --
// far fewer synchronization points than the basic one-kernel-per-cell-
// anti-diagonal approach in 02_smith_waterman.cu. Each tile is staged in
// shared memory and written back to global memory in a coalesced manner by
// store_tile (Fig. 16.12) -- a corner-turning-style use of shared memory
// (cf. Chapter 6 / Chapter 12's packing technique).
//
// Per this project's convention, every chapter file is self-contained: this
// file does NOT include or reference 01_floyd_warshall.cu or
// 02_smith_waterman.cu. The CPU reference, sequence generator, and MATCH/
// MISMATCH/INSERTION/DELETION constants are duplicated locally, matching
// 02_smith_waterman.cu's values.
//
// The book's kernel signature uses a template parameter T for the sequence
// element type (never shown declared in the printed figures); this file
// instantiates it concretely as `char`, matching the DNA-base-pair alphabet
// used throughout §16.5. Everything else below (variable names, structure,
// line-by-line logic) reproduces Figs. 16.8-16.12 as printed.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <random>
#include <algorithm>

#include "../../common/cuda_utils.h"

#define MATCH 3
#define MISMATCH (-3)
#define INSERTION (-2)
#define DELETION (-2)

// ---------------------------------------------------------------------------
// §16.6, Fig. 16.10: device functions to load the north/west/northwest
// neighbor cells a tile's anti-diagonal cell depends on -- from shared
// memory if the neighbor is inside the current tile, otherwise from global
// memory (it was produced by a previous kernel call, i.e. a previous tile
// anti-diagonal d).
// ---------------------------------------------------------------------------
__device__ inline int load_n(int *sw, int r, int q, unsigned int L, int *swTile, int r_tile, int q_tile,
                              int tile_width) {
    return (r_tile == 0) ? sw[(r - 1) * L + q] : swTile[(r_tile - 1) * tile_width + q_tile];
}

__device__ inline int load_w(int *sw, int r, int q, unsigned int L, int *swTile, int r_tile, int q_tile,
                              int tile_width) {
    return (q_tile == 0) ? sw[r * L + (q - 1)] : swTile[r_tile * tile_width + (q_tile - 1)];
}

__device__ inline int load_nw(int *sw, int r, int q, unsigned int L, int *swTile, int r_tile, int q_tile,
                               int tile_width) {
    return (r_tile == 0 || q_tile == 0) ? sw[(r - 1) * L + (q - 1)] : swTile[(r_tile - 1) * tile_width + (q_tile - 1)];
}

// §16.6, Fig. 16.11: maximum of four integers.
__device__ inline int max4(int a, int b, int c, int d) {
    int m = a;
    if (m < b) m = b;
    if (m < c) m = c;
    if (m < d) m = d;
    return m;
}

// §16.6, Fig. 16.12: write a completed tile from shared memory back to
// global memory, row by row, with all threads collaborating for coalesced
// accesses.
__device__ inline void store_tile(int *sw, int *swTile, unsigned int L, int tile_width, int tile_row, int tile_col) {
    for (int row = 0; row < tile_width; row++) {
        int r = tile_width * tile_row + row + 1;
        int q = tile_width * tile_col + threadIdx.x + 1;
        if (r < (int)L && q < (int)L) sw[r * L + q] = swTile[row * tile_width + threadIdx.x];
    }
}

// ---------------------------------------------------------------------------
// §16.6, Fig. 16.9: block-level tiling kernel. Each thread block of
// tile_width threads computes one tile_width x tile_width tile of the
// scoring matrix `sw`, iterating over the tile's own anti-diagonals
// (d_tile) and locally synchronizing after each one.
// ---------------------------------------------------------------------------
__global__ void sw_kernel_square(int *sw, char *rea, char *ref, unsigned int L, unsigned int d) {
    extern __shared__ int swTile[];
    const int tile_width = blockDim.x;
    const int numTiles_x = (L - 1 + tile_width - 1) / tile_width;
    // Tile indices
    const int tile_row = blockIdx.x;
    const int tile_col = d - blockIdx.x;
    if (tile_col >= 0 && tile_col < numTiles_x) {
        // Iterate over anti-diagonals of the tile
        for (int d_tile = 0; d_tile < 2 * tile_width - 1; d_tile++) {
            // Row indices in tile and global memory
            int r_tile = threadIdx.x;
            int r = tile_width * tile_row + r_tile + 1;
            // Column indices in tile and global memory
            int q_tile = d_tile - threadIdx.x;
            int q = tile_width * tile_col + q_tile + 1;
            // Bound checking
            if (q_tile >= 0 && q_tile < tile_width && r < (int)L && q < (int)L) {
                // Load from the previous two anti-diagonals
                int n = load_n(sw, r, q, L, swTile, r_tile, q_tile, tile_width);
                int w = load_w(sw, r, q, L, swTile, r_tile, q_tile, tile_width);
                int nw = load_nw(sw, r, q, L, swTile, r_tile, q_tile, tile_width);
                // Similarity score
                int subs_val = (rea[r - 1] == ref[q - 1]) ? MATCH : MISMATCH;
                // Obtain maximum and store in shared memory
                swTile[r_tile * tile_width + q_tile] = max4(0, nw + subs_val, w + DELETION, n + INSERTION);
            }
            __syncthreads();  // Thread block synchronization
        }
        // Store the tile in global memory
        store_tile(sw, swTile, L, tile_width, tile_row, tile_col);
    }
}

// ---------------------------------------------------------------------------
// CPU reference (duplicated locally, matching 02_smith_waterman.cu's
// recurrence and constants): straightforward row-major DP fill.
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
// §16.6, Fig. 16.8: host code -- one kernel launch per tile anti-diagonal d,
// 0 .. 2*numTiles_x-2 (2*numTiles_x-1 launches total).
// ---------------------------------------------------------------------------
float runSmithWatermanTiledGPU(std::vector<int> &H_h, const std::string &seqR, const std::string &seqC, int L_seq,
                                int threads) {
    // Length of scoring matrix side
    int L = L_seq + 1;
    size_t hBytes = static_cast<size_t>(L) * L * sizeof(int);

    int *sw_d;
    char *rea_d, *ref_d;
    CUDA_CHECK(cudaMalloc(&sw_d, hBytes));
    CUDA_CHECK(cudaMalloc(&rea_d, L_seq));
    CUDA_CHECK(cudaMalloc(&ref_d, L_seq));
    CUDA_CHECK(cudaMemcpy(sw_d, H_h.data(), hBytes, cudaMemcpyHostToDevice));  // zero-initialized boundary
    CUDA_CHECK(cudaMemcpy(rea_d, seqR.data(), L_seq, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ref_d, seqC.data(), L_seq, cudaMemcpyHostToDevice));

    // Number of tiles in x dimension
    int numTiles_x = (L_seq + threads - 1) / threads;
    // Max blocks per antidiagonal
    int numBlocks = numTiles_x;
    // threads*threads*sizeof(int) dynamic shared mem per block: at threads=128
    // this is 64KB, at or above the 48KB default per-block cap on many GPUs
    // (sm_75 included) -- would need cudaFuncAttributeMaxDynamicSharedMemorySize
    // opt-in past threads~96-128 depending on the GPU. threads<=64 here stays
    // well under the cap.
    size_t shmemBytes = static_cast<size_t>(threads) * threads * sizeof(int);

    GpuTimer timer;
    timer.start();
    // Loop over anti-diagonals of tiles
    for (unsigned int d = 0; d < 2u * numTiles_x - 1u; d++) {
        // Kernel call
        sw_kernel_square<<<numBlocks, threads, shmemBytes>>>(sw_d, rea_d, ref_d, L, d);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    float ms = timer.stopAndGetMs();

    CUDA_CHECK(cudaMemcpy(H_h.data(), sw_d, hBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(sw_d));
    CUDA_CHECK(cudaFree(rea_d));
    CUDA_CHECK(cudaFree(ref_d));
    return ms;
}

bool runTestCase(int L_seq, int threads) {
    std::string seqR = generateSequence(L_seq, 100u + static_cast<unsigned int>(L_seq));
    std::string seqC = generateSequence(L_seq, 200u + static_cast<unsigned int>(L_seq) * 7u);

    int L = L_seq + 1;
    std::vector<int> H_cpu(static_cast<size_t>(L) * L, 0);
    smithWatermanCPU(H_cpu, seqR, seqC, L_seq);

    std::vector<int> H_gpu(static_cast<size_t>(L) * L, 0);
    float ms = runSmithWatermanTiledGPU(H_gpu, seqR, seqC, L_seq, threads);

    bool ok = (H_gpu == H_cpu);
    int bestCpu = *std::max_element(H_cpu.begin(), H_cpu.end());
    int bestGpu = *std::max_element(H_gpu.begin(), H_gpu.end());
    printf("L_seq=%-5d threads=%-3d best_score(cpu)=%-4d best_score(gpu)=%-4d %.4f ms  [%s]\n", L_seq, threads,
           bestCpu, bestGpu, ms, ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    printf("Block-tiled wavefront Smith-Waterman (§16.6, Figs. 16.8-16.12):\n");
    bool ok = true;
    ok = runTestCase(1, 32) && ok;
    ok = runTestCase(7, 32) && ok;     // single tile, L_seq < tile_width
    ok = runTestCase(32, 32) && ok;    // exactly one tile
    ok = runTestCase(33, 32) && ok;    // one tile + one incomplete tile
    ok = runTestCase(256, 32) && ok;
    ok = runTestCase(300, 64) && ok;   // not a multiple of tile width

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
