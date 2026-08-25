// Chapter 16: Dynamic programming and wavefront parallelism
// §16.7, Figs. 16.14-16.19: hyperplane transformation (hypertiles) applied
// to Smith-Waterman's block-tiled wavefront kernel.
//
// Square tiles (§16.6) have two drawbacks (Fig. 16.13): non-uniform
// anti-diagonal lengths within a tile (many idle threads per iteration),
// and poor cache locality across neighboring tiles. §16.7 fixes both with a
// shear transformation ("hyperplane partitioning"): shifting row r of a
// tile_width x tile_width tile left by r cells turns each square tile into
// a parallelogram "hypertile" where every column of the original tile
// becomes an anti-diagonal after the transform (Fig. 16.14). This makes
// every intra-tile wavefront the same length (tile_width, so tile_width
// iterations instead of 2*tile_width-1) and lets the last anti-diagonal of
// one tile flow directly into the first anti-diagonal of the next.
//
// The transform is: a cell with tile-level coordinates (r_tile, q_tile),
// which under square tiling maps to score-matrix coordinates (r, q), maps
// instead to (r, q + m * r_tile) after shearing, with shear factor m = -1
// (the _shear() macro, Fig. 16.16 line 2).
//
// This file reimplements Figs. 16.15 (host code), 16.16 (kernel
// sw_kernel_hyper), 16.17 (load_n/load_w/load_nw), 16.18 (store_tile), and
// 16.19 (initialize_tile) as printed. As with 03_wavefront_block_tiled.cu,
// the book's T sequence-element type is instantiated concretely as `char`.
// Per this project's self-contained-file convention, the CPU reference,
// sequence generator, and MATCH/MISMATCH/INSERTION/DELETION constants are
// duplicated locally (matching 02/03's values) rather than referencing
// either of those files.
//
// Scope note: §16.7's closing paragraphs (after Fig. 16.20) describe an
// additional shared-memory *padding* optimization (a pad(x) macro) to avoid
// bank conflicts when the hypertile width divides the bank count evenly.
// That discussion states the padded allocation would replace line 11 of
// Fig. 16.15, but no updated code figure for the kernel/device functions
// with padded indexing throughout is shown -- unlike Figs. 16.15-16.19
// above, it is prose-described rather than a printed listing, so (per this
// project's scope rules) it is not implemented here.

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

#define _m (-1)                     // Shear factor
#define _shear(x, y) (x + _m * y)   // Affine transformation

// ---------------------------------------------------------------------------
// §16.7, Fig. 16.17: load north/west/northwest neighbors, adjusted for the
// shear transformation's effect on where a value lives within the square
// shared-memory storage of a hypertile.
// ---------------------------------------------------------------------------
__device__ inline int load_n(int *sw, int r, int q, unsigned int L, int *swTile, int r_tile, int q_tile,
                              int tile_width) {
    return (r_tile == 0 || q_tile == 0) ? sw[(r - 1) * L + q] : swTile[(r_tile - 1) * tile_width + q_tile - 1];
}

__device__ inline int load_w(int *sw, int r, int q, unsigned int L, int *swTile, int r_tile, int q_tile,
                              int tile_width) {
    return (q_tile == 0) ? sw[r * L + (q - 1)] : swTile[r_tile * tile_width + (q_tile - 1)];
}

__device__ inline int load_nw(int *sw, int r, int q, unsigned int L, int *swTile, int r_tile, int q_tile,
                               int tile_width) {
    return (r_tile == 0 || q_tile == 0 || q_tile == 1) ? sw[(r - 1) * L + (q - 1)]
                                                         : swTile[(r_tile - 1) * tile_width + (q_tile - 2)];
}

// Same as 03_wavefront_block_tiled.cu's max4 (Fig. 16.11).
__device__ inline int max4(int a, int b, int c, int d) {
    int m = a;
    if (m < b) m = b;
    if (m < c) m = c;
    if (m < d) m = d;
    return m;
}

// §16.7, Fig. 16.18: store a hypertile to global memory, applying the shear
// transformation to the column index.
__device__ inline void store_tile(int *sw, int *swTile, unsigned int L, int tile_width, int tile_row, int tile_col) {
    for (unsigned int row = 0; row < (unsigned int)tile_width; row++) {
        int r = tile_width * tile_row + row + 1;
        int q = tile_width * tile_col + _shear((int)threadIdx.x, (int)row) + 1;
        if (r < (int)L && q < (int)L) sw[r * L + q] = swTile[row * tile_width + threadIdx.x];
    }
}

// §16.7, Fig. 16.19: zero-initialize the shared memory tile so out-of-bound
// threads (whose cell falls outside the scoring matrix but who stay active
// through all iterations, unlike the square-tile kernel) never read garbage
// via load_n/load_w/load_nw.
__device__ inline void initialize_tile(int *swTile, int tile_width) {
    for (unsigned int row = 0; row < (unsigned int)tile_width; row++) {
        swTile[row * tile_width + threadIdx.x] = 0;
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------
// §16.7, Fig. 16.16: hypertile kernel. tile_width anti-diagonals per tile
// (all threads active in every iteration, unlike the square-tile kernel's
// 2*tile_width-1 iterations with a growing/shrinking active count).
// ---------------------------------------------------------------------------
__global__ void sw_kernel_hyper(int *sw, char *rea, char *ref, unsigned int L, unsigned int d) {
    extern __shared__ int swTile[];
    const int tile_width = blockDim.x;
    const int numTiles_x = (L - 1 + tile_width - 1) / tile_width;
    // Tile indices
    const int tile_row = blockIdx.x;
    const int tile_col = d - blockIdx.x * 2;
    if (tile_col >= 0 && tile_col <= numTiles_x) {
        initialize_tile(swTile, tile_width);
        // Iterate over anti-diagonals of the tile
        for (int d_tile = 0; d_tile < tile_width; d_tile++) {
            // Row indices in tile and global memory
            int r_tile = threadIdx.x;
            int r = tile_width * tile_row + r_tile + 1;
            // Column indices in tile and global memory
            int q_tile = d_tile;
            int q = tile_width * tile_col + _shear(q_tile, r_tile) + 1;
            // Bound checking
            if (r < (int)L && q >= 1 && q < (int)L) {
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
// CPU reference (duplicated locally, matching 02/03's recurrence and
// constants): the hypertile schedule computes the exact same H matrix, just
// in a different cell order.
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
// §16.7, Fig. 16.15: host code -- one kernel launch per tile anti-diagonal
// d, 0 .. 3*numTiles_x-2 (3*numTiles_x-1 launches total, more than the
// square tile version's 2*numTiles_x-1, per §16.7's iteration-count
// analysis).
// ---------------------------------------------------------------------------
float runSmithWatermanHyperGPU(std::vector<int> &H_h, const std::string &seqR, const std::string &seqC, int L_seq,
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
    // Blocks per antidiagonal
    unsigned int numBlocks = numTiles_x;
    size_t shmemBytes = static_cast<size_t>(threads) * threads * sizeof(int);

    GpuTimer timer;
    timer.start();
    // Loop over anti-diagonals of tiles
    for (unsigned int d = 0; d < 3u * numTiles_x - 1u; d++) {
        // Kernel call
        sw_kernel_hyper<<<numBlocks, threads, shmemBytes>>>(sw_d, rea_d, ref_d, L, d);
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
    float ms = runSmithWatermanHyperGPU(H_gpu, seqR, seqC, L_seq, threads);

    bool ok = (H_gpu == H_cpu);
    int bestCpu = *std::max_element(H_cpu.begin(), H_cpu.end());
    int bestGpu = *std::max_element(H_gpu.begin(), H_gpu.end());
    printf("L_seq=%-5d threads=%-3d best_score(cpu)=%-4d best_score(gpu)=%-4d %.4f ms  [%s]\n", L_seq, threads,
           bestCpu, bestGpu, ms, ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    printf("Hypertile (hyperplane-transformed) wavefront Smith-Waterman (§16.7, Figs. 16.15-16.19):\n");
    bool ok = true;
    ok = runTestCase(1, 32) && ok;
    ok = runTestCase(32, 32) && ok;    // exactly one tile
    ok = runTestCase(33, 32) && ok;    // one tile + one incomplete tile
    ok = runTestCase(256, 32) && ok;
    ok = runTestCase(300, 64) && ok;   // not a multiple of tile width

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
