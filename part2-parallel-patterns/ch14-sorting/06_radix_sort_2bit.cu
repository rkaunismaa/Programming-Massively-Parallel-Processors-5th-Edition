// Chapter 14: Sorting
// §14.7  Choice of radix value -- 2-bit radix sort with coalescing
//        (Figs. 14.10-14.12)
//
// §14.7 generalizes the 1-bit radix sort (files 03/04) to a 2-bit radix:
// "Fig. 14.10 shows an example of how radix sort can be performed using a
// 2-bit radix. Each iteration uses two bits to distribute the keys to
// buckets. Hence, the 4-bit keys can be fully sorted using only two
// iterations" (half as many iterations as a 1-bit radix, for the same key
// width). §14.7 gives three complete worked figures for this -- Fig. 14.10
// (the 2-bit sort itself), Fig. 14.11 (parallelizing + coalescing it via
// local shared-memory sort, the direct 2-bit analog of Fig. 14.8), and
// Fig. 14.12 (finding each block's 4 local buckets' global destinations,
// the direct analog of Fig. 14.9) -- before closing with "We leave the
// implementation of radix sort with a multi-bit radix as an exercise for
// the reader." Per this project's established convention for this exact
// chapter (see the "note on scope" at the top of the README: files 02/04/05
// already implement techniques given this identical treatment), the
// mechanism here is described and illustrated in full, so it's in scope;
// only the numbered §14.11 exercises list (multi-bit radix isn't one of
// them, notably) and the unillustrated decoupled-lookback-style extensions
// elsewhere in this project are treated as out of scope.
//
// §14.7's own local-sort recipe: "For the local sort inside each thread
// block, a 2-bit radix sort is performed by applying two consecutive 1-bit
// radix sort iterations. Each of these 1-bit iterations requires its own
// exclusive scan operation, however, these operations are local to the
// thread block so there is no coordination across thread blocks in between
// the two 1-bit iterations." Following the chapter's own LSD (least-
// significant-digit) convention, the lower of the two bits is partitioned
// first, then the higher bit -- chaining two stable 1-bit partitions this
// way is exactly why LSD radix sort works at the top level across
// iterations, applied here one level down, within a single block's tile.
//
// Structure mirrors file 04 exactly, generalized from 2 buckets to 4:
//   1. localSortAndCountKernel2Bit (Fig. 14.11): two chained local 1-bit
//      partitions in shared memory, staged back to global memory at the
//      same (coalesced) tile offset, plus each block's 4 local bucket
//      sizes recorded into a 4*numBlocks counts table (Fig. 14.12's
//      layout: row-major, bucket-major, block-minor).
//   2. A small host-side exclusive scan over that table (still tiny
//      relative to n) gives each block's 4 global bucket offsets.
//   3. scatterToGlobalKernel2Bit (Fig. 14.12's destination indices): each
//      block recovers its locally-sorted tile, and each thread computes
//      its bucket and rank-within-bucket to scatter to the right global
//      offset -- coalesced within each bucket sub-range, same pattern as
//      file 04's scatter.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#include "../../common/cuda_utils.h"

#define BLOCK_DIM 256
#define NUM_BITS 16          // keys are drawn from [0, 2^NUM_BITS)
#define RADIX_BITS 2
#define NUM_BUCKETS 4

// ---------------------------------------------------------------------------
// §14.7, Fig. 14.11: local 2-bit radix partition via two chained 1-bit
// passes, staged back to global memory in coalesced, tile-sequential order.
// Also emits each block's 4 local bucket sizes for the host-side scan step.
// ---------------------------------------------------------------------------
__global__ void localSortAndCountKernel2Bit(const unsigned int *input, unsigned int *staged, unsigned int *counts,
                                             int n, int numBlocks, int iter) {
    __shared__ unsigned int s_keysA[BLOCK_DIM];
    __shared__ unsigned int s_keysB[BLOCK_DIM];
    __shared__ unsigned int s_bits[BLOCK_DIM];

    int tid = threadIdx.x;
    int blockStart = blockIdx.x * blockDim.x;
    int remaining = n - blockStart;
    int validCount = remaining < (int)blockDim.x ? (remaining < 0 ? 0 : remaining) : (int)blockDim.x;

    unsigned int key = 0;
    if (tid < validCount) {
        key = input[blockStart + tid];  // coalesced load
    }
    s_keysA[tid] = key;
    __syncthreads();

    // Pass 1: partition by the LOWER of the two bits (LSD order).
    unsigned int bit0 = (tid < validCount) ? ((key >> (iter * RADIX_BITS)) & 1u) : 0u;
    s_bits[tid] = bit0;
    __syncthreads();
    for (unsigned int stride = 1; stride < blockDim.x; stride <<= 1) {
        unsigned int addend = 0;
        if (tid >= stride) addend = s_bits[tid - stride];
        __syncthreads();
        if (tid >= stride) s_bits[tid] += addend;
        __syncthreads();
    }
    unsigned int ones0Total = (validCount > 0) ? s_bits[validCount - 1] : 0u;
    unsigned int ones0Before = s_bits[tid] - bit0;
    if (tid < validCount) {
        unsigned int dst0 = (bit0 == 0) ? (unsigned int)(tid - ones0Before)
                                         : (unsigned int)(validCount - ones0Total + ones0Before);
        s_keysB[dst0] = key;
    }
    __syncthreads();

    // Pass 2: partition the pass-1 result by the HIGHER of the two bits.
    // Stability of both passes is what makes chaining them correct: within
    // each bit1-bucket, the bit0-order from pass 1 is preserved, giving the
    // fully sorted 00,01,10,11 order overall.
    unsigned int key2 = (tid < validCount) ? s_keysB[tid] : 0u;
    unsigned int bit1 = (tid < validCount) ? ((key2 >> (iter * RADIX_BITS + 1)) & 1u) : 0u;
    s_bits[tid] = bit1;
    __syncthreads();
    for (unsigned int stride = 1; stride < blockDim.x; stride <<= 1) {
        unsigned int addend = 0;
        if (tid >= stride) addend = s_bits[tid - stride];
        __syncthreads();
        if (tid >= stride) s_bits[tid] += addend;
        __syncthreads();
    }
    unsigned int ones1Total = (validCount > 0) ? s_bits[validCount - 1] : 0u;
    unsigned int ones1Before = s_bits[tid] - bit1;
    if (tid < validCount) {
        unsigned int dst1 = (bit1 == 0) ? (unsigned int)(tid - ones1Before)
                                         : (unsigned int)(validCount - ones1Total + ones1Before);
        s_keysA[dst1] = key2;  // reuse s_keysA as the final, fully 2-bit-sorted tile
    }
    __syncthreads();

    if (tid < validCount) {
        staged[blockStart + tid] = s_keysA[tid];  // coalesced store, same tile offset as the load
    }

    // Count the 4 buckets directly from the now-sorted tile. A single
    // thread does this serially (validCount <= BLOCK_DIM elements): simple
    // and easy to verify correct, and not on the critical (coalescing)
    // path this file is actually demonstrating.
    if (tid == 0) {
        unsigned int c[NUM_BUCKETS] = {0u, 0u, 0u, 0u};
        for (int k = 0; k < validCount; ++k) {
            unsigned int bucket = (s_keysA[k] >> (iter * RADIX_BITS)) & (NUM_BUCKETS - 1u);
            ++c[bucket];
        }
        for (int b = 0; b < NUM_BUCKETS; ++b) {
            counts[b * numBlocks + blockIdx.x] = c[b];
        }
    }
}

// ---------------------------------------------------------------------------
// §14.7, Fig. 14.12: scatter each block's staged, locally-sorted tile to its
// global bucket positions. Each thread recovers its bucket directly from its
// (already locally-sorted) key, and its rank within that bucket from its
// tile position minus the block-local prefix count of earlier buckets.
// ---------------------------------------------------------------------------
__global__ void scatterToGlobalKernel2Bit(const unsigned int *staged, unsigned int *output, const unsigned int *counts,
                                           const unsigned int *offsets, int n, int numBlocks, int iter) {
    int tid = threadIdx.x;
    int blockStart = blockIdx.x * blockDim.x;
    int remaining = n - blockStart;
    int validCount = remaining < (int)blockDim.x ? (remaining < 0 ? 0 : remaining) : (int)blockDim.x;
    if (tid >= validCount) return;

    unsigned int c0 = counts[0 * numBlocks + blockIdx.x];
    unsigned int c1 = counts[1 * numBlocks + blockIdx.x];
    unsigned int c2 = counts[2 * numBlocks + blockIdx.x];
    unsigned int localPrefix[NUM_BUCKETS] = {0u, c0, c0 + c1, c0 + c1 + c2};

    unsigned int key = staged[blockStart + tid];
    unsigned int bucket = (key >> (iter * RADIX_BITS)) & (NUM_BUCKETS - 1u);
    unsigned int rankInBucket = (unsigned int)tid - localPrefix[bucket];
    unsigned int globalBase = offsets[bucket * numBlocks + blockIdx.x];
    output[globalBase + rankInBucket] = key;
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

float runRadixSort2Bit(const std::vector<unsigned int> &input_h, std::vector<unsigned int> &out_h) {
    int n = static_cast<int>(input_h.size());
    size_t bytes = n * sizeof(unsigned int);
    int numBlocks = (n + BLOCK_DIM - 1) / BLOCK_DIM;
    size_t tableBytes = NUM_BUCKETS * numBlocks * sizeof(unsigned int);

    unsigned int *bufA_d, *bufB_d, *staged_d, *counts_d, *offsets_d;
    CUDA_CHECK(cudaMalloc(&bufA_d, bytes));
    CUDA_CHECK(cudaMalloc(&bufB_d, bytes));
    CUDA_CHECK(cudaMalloc(&staged_d, bytes));
    CUDA_CHECK(cudaMalloc(&counts_d, tableBytes));
    CUDA_CHECK(cudaMalloc(&offsets_d, tableBytes));

    std::vector<unsigned int> counts_h(NUM_BUCKETS * numBlocks), offsets_h(NUM_BUCKETS * numBlocks);

    auto resetInput = [&]() {
        CUDA_CHECK(cudaMemcpy(bufA_d, input_h.data(), bytes, cudaMemcpyHostToDevice));
    };

    dim3 block(BLOCK_DIM);
    dim3 grid(numBlocks);
    int iterations = NUM_BITS / RADIX_BITS;

    auto sortPass = [&]() -> unsigned int * {
        unsigned int *cur = bufA_d, *nxt = bufB_d;
        for (int iter = 0; iter < iterations; ++iter) {
            localSortAndCountKernel2Bit<<<grid, block>>>(cur, staged_d, counts_d, n, numBlocks, iter);
            CUDA_CHECK(cudaGetLastError());

            // §14.7, Fig. 14.12: small host-side exclusive scan over the
            // NUM_BUCKETS*numBlocks-element bucket-size table (row-major:
            // bucket-major, block-minor, matching Fig. 14.12's layout).
            CUDA_CHECK(cudaMemcpy(counts_h.data(), counts_d, tableBytes, cudaMemcpyDeviceToHost));
            unsigned int running = 0;
            for (int t = 0; t < NUM_BUCKETS * numBlocks; ++t) {
                offsets_h[t] = running;
                running += counts_h[t];
            }
            CUDA_CHECK(cudaMemcpy(offsets_d, offsets_h.data(), tableBytes, cudaMemcpyHostToDevice));

            scatterToGlobalKernel2Bit<<<grid, block>>>(staged_d, nxt, counts_d, offsets_d, n, numBlocks, iter);
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
    std::vector<unsigned int> input_h = generateInput(n, 24680u + static_cast<unsigned int>(n));

    std::vector<unsigned int> ref = input_h;
    std::sort(ref.begin(), ref.end());

    std::vector<unsigned int> gpu_h;
    float ms = runRadixSort2Bit(input_h, gpu_h);

    int numBlocks = (n + BLOCK_DIM - 1) / BLOCK_DIM;
    bool ok = (gpu_h == ref);
    printf("n=%d (%d bits, radix=2^%d, %d blocks of %d, %d iterations): %.4f ms  [%s]\n", n, NUM_BITS,
           RADIX_BITS, numBlocks, BLOCK_DIM, NUM_BITS / RADIX_BITS, ms, ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    printf("Radix sort with a 2-bit radix (§14.7, Figs. 14.10-14.12):\n");
    bool ok = true;
    ok = runTestCase(1) && ok;
    ok = runTestCase(1000) && ok;    // last block partial
    ok = runTestCase(1 << 16) && ok;
    ok = runTestCase(70000) && ok;   // last block partial, many blocks

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
