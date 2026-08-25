// Chapter 14: Sorting
// §14.6, Figs. 14.8-14.9: radix sort iteration optimized for memory
// coalescing, extended to many thread blocks.
//
// File 03's single-block kernel writes each key directly to its
// (data-dependent) global destination -- adjacent threads generally do NOT
// write to adjacent addresses, so those writes cannot be coalesced. §14.6
// fixes this with the third of the book's three general coalescing
// strategies (rearrange threads / rearrange data / stage through shared
// memory): each block first sorts its OWN tile locally into shared memory
// (zero-bucket keys first, then one-bucket keys -- a local 1-bit radix
// partition, computed exactly like file 03's kernel but block-scoped), then
// copies that local result to global memory in strict thread-index order,
// which is fully coalesced. This is done in two kernels plus a tiny host
// step, mirroring Figs. 14.8-14.9:
//
//   1. localSortAndCountKernel (Fig. 14.8): each block loads its
//      BLOCK_DIM-key tile (coalesced), does a block-wide exclusive scan of
//      the bits to find each key's LOCAL destination within the tile,
//      reorders the tile in shared memory, then writes the reordered tile
//      back to a `staged` global array at the SAME tile offset it read from
//      (again coalesced -- the reorder happened entirely in shared memory).
//      It also records each block's local bucket sizes (# zeros, # ones)
//      into a `counts` table, laid out row-major exactly as Fig. 14.9
//      shows: all blocks' zero-bucket counts, followed by all blocks'
//      one-bucket counts.
//   2. An exclusive scan over that (small, 2*numBlocks-element) `counts`
//      table gives each block's GLOBAL starting offset for its zero bucket
//      and its one bucket (Fig. 14.9). Because this table is tiny compared
//      to n, this file scans it on the host rather than pulling in Chapter
//      11's grid-wide scan kernel (see ch11-scan/06_scan_global_multi_block.cu
//      for that machinery) -- staying self-contained and focused on the
//      coalescing technique itself.
//   3. scatterToGlobalKernel: each block reads back its own staged,
//      locally-sorted tile and writes the zero-bucket keys starting at its
//      global zero offset and the one-bucket keys starting at its global
//      one offset. Within each bucket, consecutive threads write consecutive
//      addresses -- exactly the coalesced pattern Fig. 14.8's text describes
//      ("the first two threads both write to adjacent locations... while
//      the last two threads also write to adjacent locations").

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#include "../../common/cuda_utils.h"

#define BLOCK_DIM 256
#define NUM_BITS 16  // keys are drawn from [0, 2^NUM_BITS)

// ---------------------------------------------------------------------------
// §14.6, Fig. 14.8: local (block-wide) 1-bit radix partition, staged back
// to global memory in coalesced, tile-sequential order. Also emits each
// block's local bucket sizes for the host-side scan step.
// ---------------------------------------------------------------------------
__global__ void localSortAndCountKernel(const unsigned int *input, unsigned int *staged, unsigned int *counts,
                                         int n, int numBlocks, int iter) {
    __shared__ unsigned int s_bits[BLOCK_DIM];
    __shared__ unsigned int s_sorted[BLOCK_DIM];

    int tid = threadIdx.x;
    int blockStart = blockIdx.x * blockDim.x;
    int remaining = n - blockStart;
    int validCount = remaining < (int)blockDim.x ? (remaining < 0 ? 0 : remaining) : (int)blockDim.x;

    unsigned int key = 0, bit = 0;
    if (tid < validCount) {
        key = input[blockStart + tid];  // coalesced load
        bit = (key >> iter) & 1u;
    }
    s_bits[tid] = bit;
    __syncthreads();

    // Block-wide Hillis-Steele inclusive scan of the bits.
    for (unsigned int stride = 1; stride < blockDim.x; stride <<= 1) {
        unsigned int addend = 0;
        if (tid >= stride) addend = s_bits[tid - stride];
        __syncthreads();
        if (tid >= stride) s_bits[tid] += addend;
        __syncthreads();
    }

    unsigned int localOnesTotal = (validCount > 0) ? s_bits[validCount - 1] : 0u;
    unsigned int numOnesBefore = s_bits[tid] - bit;

    if (tid < validCount) {
        unsigned int localDst =
            (bit == 0) ? (unsigned int)(tid - numOnesBefore) : (validCount - localOnesTotal + numOnesBefore);
        s_sorted[localDst] = key;
    }
    __syncthreads();

    if (tid < validCount) {
        staged[blockStart + tid] = s_sorted[tid];  // coalesced store, same tile offset as the load
    }
    if (tid == 0) {
        counts[0 * numBlocks + blockIdx.x] = validCount - localOnesTotal;  // local zero-bucket size
        counts[1 * numBlocks + blockIdx.x] = localOnesTotal;               // local one-bucket size
    }
}

// ---------------------------------------------------------------------------
// §14.6, Fig. 14.9: scatter each block's staged, locally-sorted tile to its
// global bucket positions. Threads within the zero range write consecutive
// addresses starting at offsets[zero bucket, this block]; threads within
// the one range write consecutive addresses starting at offsets[one bucket,
// this block]. Both sub-writes are coalesced.
// ---------------------------------------------------------------------------
__global__ void scatterToGlobalKernel(const unsigned int *staged, unsigned int *output, const unsigned int *counts,
                                       const unsigned int *offsets, int n, int numBlocks) {
    int tid = threadIdx.x;
    int blockStart = blockIdx.x * blockDim.x;
    int remaining = n - blockStart;
    int validCount = remaining < (int)blockDim.x ? (remaining < 0 ? 0 : remaining) : (int)blockDim.x;
    if (tid >= validCount) return;

    unsigned int localZeros = counts[0 * numBlocks + blockIdx.x];
    unsigned int zeroStart = offsets[0 * numBlocks + blockIdx.x];
    unsigned int oneStart = offsets[1 * numBlocks + blockIdx.x];

    unsigned int key = staged[blockStart + tid];
    unsigned int dst = ((unsigned int)tid < localZeros) ? (zeroStart + tid) : (oneStart + (tid - localZeros));
    output[dst] = key;
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

float runRadixSortCoalesced(const std::vector<unsigned int> &input_h, std::vector<unsigned int> &out_h) {
    int n = static_cast<int>(input_h.size());
    size_t bytes = n * sizeof(unsigned int);
    int numBlocks = (n + BLOCK_DIM - 1) / BLOCK_DIM;
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
            localSortAndCountKernel<<<grid, block>>>(cur, staged_d, counts_d, n, numBlocks, iter);
            CUDA_CHECK(cudaGetLastError());

            // §14.6, Fig. 14.9: small host-side exclusive scan over the
            // 2*numBlocks-element bucket-size table (row-major: all blocks'
            // zero counts, then all blocks' one counts).
            CUDA_CHECK(cudaMemcpy(counts_h.data(), counts_d, tableBytes, cudaMemcpyDeviceToHost));
            unsigned int running = 0;
            for (int t = 0; t < 2 * numBlocks; ++t) {
                offsets_h[t] = running;
                running += counts_h[t];
            }
            CUDA_CHECK(cudaMemcpy(offsets_d, offsets_h.data(), tableBytes, cudaMemcpyHostToDevice));

            scatterToGlobalKernel<<<grid, block>>>(staged_d, nxt, counts_d, offsets_d, n, numBlocks);
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
    std::vector<unsigned int> input_h = generateInput(n, 13579u + static_cast<unsigned int>(n));

    std::vector<unsigned int> ref = input_h;
    std::sort(ref.begin(), ref.end());

    std::vector<unsigned int> gpu_h;
    float ms = runRadixSortCoalesced(input_h, gpu_h);

    int numBlocks = (n + BLOCK_DIM - 1) / BLOCK_DIM;
    bool ok = (gpu_h == ref);
    printf("n=%d (%d bits, %d blocks of %d): %.4f ms  [%s]\n", n, NUM_BITS, numBlocks, BLOCK_DIM, ms,
           ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    printf("Radix sort optimized for memory coalescing (§14.6, Figs. 14.8-14.9):\n");
    bool ok = true;
    ok = runTestCase(1) && ok;
    ok = runTestCase(1000) && ok;    // last block partial
    ok = runTestCase(1 << 16) && ok;
    ok = runTestCase(70000) && ok;   // last block partial, many blocks

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
