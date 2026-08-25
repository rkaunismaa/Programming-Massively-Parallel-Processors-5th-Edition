// Chapter 14: Sorting
// §14.8, Fig. 14.13: thread coarsening applied to the coalescing-optimized
// radix sort iteration.
//
// File 04 gives each thread exactly one key, so a block of BLOCK_DIM
// threads sorts a BLOCK_DIM-key tile. §14.8 observes two costs that grow as
// tiles get smaller (i.e., as more, smaller blocks are used to cover the
// same n): local buckets per block shrink, leaving less to coalesce when
// they're written out; and the host-side scan table in file 04 has 2 rows
// per block, so more blocks means a bigger table to scan. Thread
// coarsening -- giving each thread COARSE_FACTOR keys instead of 1 -- grows
// each block's tile to BLOCK_DIM*COARSE_FACTOR keys without adding threads,
// so the same n needs COARSE_FACTOR times fewer blocks: bigger local
// buckets (more opportunity to coalesce) and a COARSE_FACTOR-times-smaller
// scan table (§14.8: "By applying thread coarsening, the number of blocks
// is reduced, thereby reducing the size of the table and the overhead of
// the exclusive scan operation").
//
// The local partition itself now needs a block-wide scan over
// TILE_SIZE = BLOCK_DIM*COARSE_FACTOR bits using only BLOCK_DIM threads,
// so each thread first does a sequential scan of its own COARSE_FACTOR-
// element chunk, then the block-wide Hillis-Steele scan runs over just the
// BLOCK_DIM per-thread totals, then an add-back pass folds each thread's
// exclusive total back into its chunk -- the same coarsened-scan structure
// as ch11-scan's register-tiled kernels, reimplemented locally here.
// Coalesced global loads/stores still use the "stride by BLOCK_DIM"
// indexing pattern (tile position p = c*BLOCK_DIM+tid), which happens to
// keep every tile position's linear order identical to its input order, so
// it is simultaneously coalesced and order-preserving.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#include "../../common/cuda_utils.h"

#define BLOCK_DIM 128
#define COARSE_FACTOR 8
#define TILE_SIZE (BLOCK_DIM * COARSE_FACTOR)
#define NUM_BITS 16  // keys are drawn from [0, 2^NUM_BITS)

// ---------------------------------------------------------------------------
// §14.8, Fig. 14.13: local (block-wide) 1-bit radix partition over a
// TILE_SIZE-key tile, with thread coarsening. Same staged-write pattern as
// file 04's localSortAndCountKernel, applied over a bigger tile.
// ---------------------------------------------------------------------------
__global__ void localSortAndCountCoarsenedKernel(const unsigned int *input, unsigned int *staged,
                                                   unsigned int *counts, int n, int numBlocks, int iter) {
    __shared__ unsigned int s_keys[TILE_SIZE];
    __shared__ unsigned int s_bits[TILE_SIZE];   // original bits, preserved
    __shared__ unsigned int s_scan[TILE_SIZE];   // inclusive scan of bits
    __shared__ unsigned int s_sorted[TILE_SIZE];
    __shared__ unsigned int s_threadSums[BLOCK_DIM];

    int tid = threadIdx.x;
    int blockStart = blockIdx.x * TILE_SIZE;
    int remaining = n - blockStart;
    int validCount = remaining < TILE_SIZE ? (remaining < 0 ? 0 : remaining) : TILE_SIZE;

    // Coalesced load: for fixed c, consecutive tid -> consecutive global
    // addresses. Tile position p = c*BLOCK_DIM+tid maps to global index
    // blockStart+p exactly, so this is also order-preserving.
    for (int c = 0; c < COARSE_FACTOR; ++c) {
        int p = c * BLOCK_DIM + tid;
        int idx = blockStart + p;
        unsigned int key = (idx < n) ? input[idx] : 0u;
        unsigned int bit = (idx < n) ? ((key >> iter) & 1u) : 0u;
        s_keys[p] = key;
        s_bits[p] = bit;
        s_scan[p] = bit;
    }
    __syncthreads();

    // Per-thread sequential inclusive scan of its own contiguous
    // COARSE_FACTOR-element chunk [tid*COARSE_FACTOR, tid*COARSE_FACTOR+COARSE_FACTOR).
    int threadSegment = tid * COARSE_FACTOR;
    for (int c = 1; c < COARSE_FACTOR; ++c) {
        s_scan[threadSegment + c] += s_scan[threadSegment + c - 1];
    }
    unsigned int threadSum = s_scan[threadSegment + COARSE_FACTOR - 1];
    s_threadSums[tid] = threadSum;
    __syncthreads();

    // Block-wide Hillis-Steele inclusive scan of the BLOCK_DIM per-thread sums.
    for (unsigned int stride = 1; stride < blockDim.x; stride <<= 1) {
        unsigned int addend = 0;
        if (tid >= stride) addend = s_threadSums[tid - stride];
        __syncthreads();
        if (tid >= stride) s_threadSums[tid] += addend;
        __syncthreads();
    }
    unsigned int prevThreads = (tid > 0) ? s_threadSums[tid - 1] : 0u;

    // Add-back pass: fold the exclusive total of all preceding threads'
    // chunks into this thread's chunk, completing the TILE_SIZE-wide scan.
    for (int c = 0; c < COARSE_FACTOR; ++c) {
        s_scan[threadSegment + c] += prevThreads;
    }
    __syncthreads();

    unsigned int totalOnes = (validCount > 0) ? s_scan[validCount - 1] : 0u;

    for (int c = 0; c < COARSE_FACTOR; ++c) {
        int p = c * BLOCK_DIM + tid;
        if (p < validCount) {
            unsigned int bit = s_bits[p];
            unsigned int numOnesBefore = s_scan[p] - bit;
            unsigned int dst =
                (bit == 0) ? (unsigned int)(p - numOnesBefore) : (validCount - totalOnes + numOnesBefore);
            s_sorted[dst] = s_keys[p];
        }
    }
    __syncthreads();

    for (int c = 0; c < COARSE_FACTOR; ++c) {
        int p = c * BLOCK_DIM + tid;
        if (p < validCount) {
            staged[blockStart + p] = s_sorted[p];  // coalesced store, same tile offset as the load
        }
    }
    if (tid == 0) {
        counts[0 * numBlocks + blockIdx.x] = validCount - totalOnes;
        counts[1 * numBlocks + blockIdx.x] = totalOnes;
    }
}

// ---------------------------------------------------------------------------
// §14.8, Fig. 14.9 (applied over the bigger tile): scatter each block's
// staged, locally-sorted tile to its global bucket positions.
// ---------------------------------------------------------------------------
__global__ void scatterToGlobalCoarsenedKernel(const unsigned int *staged, unsigned int *output,
                                                const unsigned int *counts, const unsigned int *offsets, int n,
                                                int numBlocks) {
    int tid = threadIdx.x;
    int blockStart = blockIdx.x * TILE_SIZE;
    int remaining = n - blockStart;
    int validCount = remaining < TILE_SIZE ? (remaining < 0 ? 0 : remaining) : TILE_SIZE;

    unsigned int localZeros = counts[0 * numBlocks + blockIdx.x];
    unsigned int zeroStart = offsets[0 * numBlocks + blockIdx.x];
    unsigned int oneStart = offsets[1 * numBlocks + blockIdx.x];

    for (int c = 0; c < COARSE_FACTOR; ++c) {
        int p = c * BLOCK_DIM + tid;
        if (p < validCount) {
            unsigned int key = staged[blockStart + p];
            unsigned int dst = ((unsigned int)p < localZeros) ? (zeroStart + p) : (oneStart + (p - localZeros));
            output[dst] = key;
        }
    }
}

std::vector<unsigned int> generateInput(int n, unsigned int seed) {
    std::vector<unsigned int> v(n);
    unsigned int state = seed;
    for (int i = 0; i < n; ++i) {
        state = state * 1103515245u + 12345u;
        v[i] = (state >> 8) & ((1u << NUM_BITS) - 1u);
    }
    return v;
}

float runRadixSortCoarsened(const std::vector<unsigned int> &input_h, std::vector<unsigned int> &out_h) {
    int n = static_cast<int>(input_h.size());
    size_t bytes = n * sizeof(unsigned int);
    int numBlocks = (n + TILE_SIZE - 1) / TILE_SIZE;
    size_t tableBytes = 2 * numBlocks * sizeof(unsigned int);

    unsigned int *bufA_d, *bufB_d, *staged_d, *counts_d, *offsets_d;
    CUDA_CHECK(cudaMalloc(&bufA_d, bytes));
    CUDA_CHECK(cudaMalloc(&bufB_d, bytes));
    CUDA_CHECK(cudaMalloc(&staged_d, bytes));
    CUDA_CHECK(cudaMalloc(&counts_d, tableBytes));
    CUDA_CHECK(cudaMalloc(&offsets_d, tableBytes));

    std::vector<unsigned int> counts_h(2 * numBlocks), offsets_h(2 * numBlocks);

    auto resetInput = [&]() {
        CUDA_CHECK(cudaMemcpy(bufA_d, input_h.data(), bytes, cudaMemcpyHostToDevice));
    };

    dim3 block(BLOCK_DIM);
    dim3 grid(numBlocks);

    auto sortPass = [&]() -> unsigned int * {
        unsigned int *cur = bufA_d, *nxt = bufB_d;
        for (int iter = 0; iter < NUM_BITS; ++iter) {
            localSortAndCountCoarsenedKernel<<<grid, block>>>(cur, staged_d, counts_d, n, numBlocks, iter);
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemcpy(counts_h.data(), counts_d, tableBytes, cudaMemcpyDeviceToHost));
            unsigned int running = 0;
            for (int t = 0; t < 2 * numBlocks; ++t) {
                offsets_h[t] = running;
                running += counts_h[t];
            }
            CUDA_CHECK(cudaMemcpy(offsets_d, offsets_h.data(), tableBytes, cudaMemcpyHostToDevice));

            scatterToGlobalCoarsenedKernel<<<grid, block>>>(staged_d, nxt, counts_d, offsets_d, n, numBlocks);
            CUDA_CHECK(cudaGetLastError());
            std::swap(cur, nxt);
        }
        return cur;
    };

    resetInput();
    sortPass();  // warm-up (untimed)
    CUDA_CHECK(cudaDeviceSynchronize());

    resetInput();
    GpuTimer timer;
    timer.start();
    unsigned int *result_d = sortPass();
    float ms = timer.stopAndGetMs();

    out_h.resize(n);
    CUDA_CHECK(cudaMemcpy(out_h.data(), result_d, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(bufA_d));
    CUDA_CHECK(cudaFree(bufB_d));
    CUDA_CHECK(cudaFree(staged_d));
    CUDA_CHECK(cudaFree(counts_d));
    CUDA_CHECK(cudaFree(offsets_d));
    return ms;
}

bool runTestCase(int n) {
    std::vector<unsigned int> input_h = generateInput(n, 8675309u + static_cast<unsigned int>(n));

    std::vector<unsigned int> ref = input_h;
    std::sort(ref.begin(), ref.end());

    std::vector<unsigned int> gpu_h;
    float ms = runRadixSortCoarsened(input_h, gpu_h);

    int numBlocks = (n + TILE_SIZE - 1) / TILE_SIZE;
    bool ok = (gpu_h == ref);
    printf("n=%d (%d bits, %d blocks, tile=%d, coarse=%d): %.4f ms  [%s]\n", n, NUM_BITS, numBlocks, TILE_SIZE,
           COARSE_FACTOR, ms, ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    printf("Thread-coarsened radix sort (§14.8, Fig. 14.13):\n");
    bool ok = true;
    ok = runTestCase(1) && ok;
    ok = runTestCase(1000) && ok;    // last block partial
    ok = runTestCase(1 << 16) && ok;
    ok = runTestCase(70000) && ok;   // last block partial, many blocks

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
