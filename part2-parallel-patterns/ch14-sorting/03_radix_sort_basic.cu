// Chapter 14: Sorting
// §14.4-14.5, Figs. 14.4-14.7: (LSD) radix sort, one iteration per kernel.
//
// Radix sort repeatedly distributes keys into buckets based on one digit
// (here, a 1-bit "digit") at a time, starting from the least significant
// bit, while preserving the relative order of keys that land in the same
// bucket (§14.4's "stable partition pattern"). After processing every bit,
// the keys are fully sorted. §14.5 focuses on parallelizing a SINGLE
// iteration (one bit), with the host looping over iterations sequentially
// (iterations depend on each other; only the work WITHIN an iteration is
// parallel).
//
// Fig. 14.7's kernel (reproduced here, matching the book's own line-by-line
// walkthrough in §14.5):
//   1. Each thread loads its key and extracts the iteration's bit:
//      bit = (key >> iter) & 1.
//   2. All threads collaborate on a grid-wide EXCLUSIVE scan of the bits
//      array. Since bits are 0/1, the scan result at position i is exactly
//      "# ones before i".
//   3. Each thread derives its key's destination:
//        bit == 0: destination = i - (#ones before i)
//        bit == 1: destination = n - (#ones total) + (#ones before i)
//      (both formulas are derived in §14.5's text from first principles).
//   4. Each thread stores its key at that destination in the output array.
//
// This file implements the grid-wide scan as a single BLOCK-wide
// Hillis-Steele inclusive scan in shared memory (n <= BLOCK_DIM, one block),
// which is exactly a grid-wide scan when the grid is one block -- this
// keeps the file focused on Fig. 14.7's destination-index derivation itself
// without pulling in Chapter 11's multi-block scan machinery. Files 04 and
// 05 in this directory extend this to many blocks while optimizing for
// memory coalescing (§14.6, §14.8).

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#include "../../common/cuda_utils.h"

#define BLOCK_DIM 1024
#define NUM_BITS 16  // keys are drawn from [0, 2^NUM_BITS)

// ---------------------------------------------------------------------------
// §14.5, Fig. 14.7: one radix sort iteration over bit `iter`. Single block
// (n <= blockDim.x); the block-wide scan below plays the role of the
// "grid-wide exclusive scan" the book's kernel calls out on line 10.
// ---------------------------------------------------------------------------
__global__ void radixSortIterKernel(const unsigned int *input, unsigned int *output, int n, int iter) {
    extern __shared__ unsigned int s_bits[];  // size blockDim.x

    int i = threadIdx.x;
    unsigned int key = 0, bit = 0;
    if (i < n) {
        key = input[i];
        bit = (key >> iter) & 1u;
    }
    s_bits[i] = bit;  // threads with i >= n contribute 0, harmless padding
    __syncthreads();

    // Hillis-Steele inclusive scan of s_bits over blockDim.x elements.
    for (unsigned int stride = 1; stride < blockDim.x; stride <<= 1) {
        unsigned int addend = 0;
        if (i >= stride) addend = s_bits[i - stride];
        __syncthreads();
        if (i >= stride) s_bits[i] += addend;
        __syncthreads();
    }

    // s_bits[n-1] is the inclusive sum over exactly the n valid elements
    // (padding beyond index n-1 never contributes to an earlier prefix).
    unsigned int numOnesTotal = s_bits[n - 1];
    unsigned int numOnesBefore = s_bits[i] - bit;  // exclusive count at i

    if (i < n) {
        unsigned int dst = (bit == 0) ? (unsigned int)(i - numOnesBefore) : (n - numOnesTotal + numOnesBefore);
        output[dst] = key;
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

float runRadixSortBasic(const std::vector<unsigned int> &input_h, std::vector<unsigned int> &out_h) {
    int n = static_cast<int>(input_h.size());
    size_t bytes = n * sizeof(unsigned int);

    unsigned int *bufA_d, *bufB_d;
    CUDA_CHECK(cudaMalloc(&bufA_d, bytes));
    CUDA_CHECK(cudaMalloc(&bufB_d, bytes));

    size_t shmemBytes = BLOCK_DIM * sizeof(unsigned int);

    auto resetInput = [&]() {
        CUDA_CHECK(cudaMemcpy(bufA_d, input_h.data(), bytes, cudaMemcpyHostToDevice));
    };

    auto sortPass = [&]() -> unsigned int * {
        unsigned int *cur = bufA_d, *nxt = bufB_d;
        for (int iter = 0; iter < NUM_BITS; ++iter) {
            radixSortIterKernel<<<1, BLOCK_DIM, shmemBytes>>>(cur, nxt, n, iter);
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
    return ms;
}

bool runTestCase(int n) {
    std::vector<unsigned int> input_h = generateInput(n, 24601u + static_cast<unsigned int>(n));

    std::vector<unsigned int> ref = input_h;
    std::sort(ref.begin(), ref.end());

    std::vector<unsigned int> gpu_h;
    float ms = runRadixSortBasic(input_h, gpu_h);

    bool ok = (gpu_h == ref);
    printf("n=%d (%d bits, single block): %.4f ms  [%s]\n", n, NUM_BITS, ms, ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    printf("Parallel radix sort, one thread per key (§14.5, Fig. 14.7):\n");
    bool ok = true;
    ok = runTestCase(1) && ok;
    ok = runTestCase(777) && ok;   // odd n
    ok = runTestCase(1000) && ok;
    ok = runTestCase(1024) && ok;  // exactly one full block

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
