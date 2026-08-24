// Chapter 13: Merge
// §13.6  A tiled merge kernel to improve coalescing (Figs. 13.10-13.14)
//
// The basic kernel (file 01) is correct but memory-inefficient: adjacent
// threads in a warp read/write scattered A/B/C addresses (their input
// sub-arrays are wherever co_rank() data-dependently placed them), and the
// co_rank() binary searches themselves hit global memory with an irregular
// access pattern. §13.6's fix: do the co-rank + merge work at BLOCK
// granularity first, cooperatively load that block's whole input slice
// into shared memory with coalesced accesses (consecutive threads load
// consecutive addresses), then let individual threads run their own
// (much cheaper) co_rank() + merge over the SHARED-memory copy.
//
// Because a block's input slice can be larger than shared memory can hold,
// the kernel iterates in "tile_size"-element chunks (Figs. 13.12-13.13):
// each while-loop iteration loads up to tile_size A elements and tile_size
// B elements into A_S/B_S, generates as many C elements as it can from
// that data, then reloads the NEXT tile starting fresh at A_S[0]/B_S[0]
// (discarding whatever of the current tile went unused -- this wasted
// bandwidth is exactly what file 03's circular buffer fixes).
//
// Book's own worked numeric example (m=33000, n=31000, gridDim.x=16,
// blockDim.x=128, tile_size=1024) is reproduced verbatim as one of this
// file's test cases below, specifically because it forces 4 while-loop
// iterations per block with a genuinely partial last tile (4*1024=4096 >
// 4000 = C elements/block) -- the exact edge case §13.6's text calls out
// as needing careful bounds handling.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#include "../../common/cuda_utils.h"

// Re-declared identically to file 01 (each file in this chapter is
// self-contained per this project's convention).
__host__ __device__ int co_rank(int k, const int *A, int m, const int *B, int n) {
    int i = k < m ? k : m;
    int j = k - i;
    int i_low = 0 > (k - n) ? 0 : (k - n);
    int j_low = 0 > (k - m) ? 0 : (k - m);
    int delta;
    bool active = true;
    while (active) {
        if (i > 0 && j < n && A[i - 1] > B[j]) {
            delta = (i - i_low + 1) >> 1;
            j_low = j;
            j += delta;
            i -= delta;
        } else if (j > 0 && i < m && B[j - 1] >= A[i]) {
            delta = (j - j_low + 1) >> 1;
            i_low = i;
            i += delta;
            j -= delta;
        } else {
            active = false;
        }
    }
    return i;
}

__host__ __device__ void merge_sequential(const int *A, int m, const int *B, int n, int *C) {
    int i = 0, j = 0, k = 0;
    while (i < m && j < n) {
        if (A[i] <= B[j]) {
            C[k++] = A[i++];
        } else {
            C[k++] = B[j++];
        }
    }
    while (i < m) C[k++] = A[i++];
    while (j < n) C[k++] = B[j++];
}

// ---------------------------------------------------------------------------
// §13.6, Figs. 13.11-13.13: tiled merge kernel.
//
// Dynamic shared memory layout: A_S = shareAB[0 .. tile_size-1],
// B_S = shareAB[tile_size .. 2*tile_size-1].
// ---------------------------------------------------------------------------
__global__ void mergeTiledKernel(const int *A, int m, const int *B, int n, int *C,
                                  int tile_size) {
    extern __shared__ int shareAB[];
    int *A_S = &shareAB[0];
    int *B_S = &shareAB[tile_size];

    int total = m + n;
    int elementsPerBlock = (total + gridDim.x - 1) / gridDim.x;
    int C_curr = blockIdx.x * elementsPerBlock;
    C_curr = C_curr < total ? C_curr : total;
    int C_next = (blockIdx.x + 1) * elementsPerBlock;
    C_next = C_next < total ? C_next : total;

    // Fig. 13.11 Part 1: one thread computes the block-level co-ranks and
    // publishes them through shared memory (A_S[0], A_S[1] reused as
    // scratch before the tiles are loaded).
    if (threadIdx.x == 0) {
        A_S[0] = co_rank(C_curr, A, m, B, n);
        A_S[1] = co_rank(C_next, A, m, B, n);
    }
    __syncthreads();
    int A_curr = A_S[0];
    int A_next = A_S[1];
    int B_curr = C_curr - A_curr;
    int B_next = C_next - A_next;
    __syncthreads();

    int C_length = C_next - C_curr;
    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;
    int total_iteration = (C_length + tile_size - 1) / tile_size;
    int C_completed = 0;
    int A_consumed = 0;
    int B_consumed = 0;

    for (int counter = 0; counter < total_iteration; ++counter) {
        // Fig. 13.12 Part 2: cooperative, coalesced load of one tile of A
        // and one tile of B (consecutive threads load consecutive
        // addresses); guarded so a thread never reads past the block's
        // remaining input elements (matters on the final, partial tile).
        for (int i = 0; i < tile_size; i += blockDim.x) {
            if (i + threadIdx.x < A_length - A_consumed) {
                A_S[i + threadIdx.x] = A[A_curr + A_consumed + i + threadIdx.x];
            }
        }
        for (int i = 0; i < tile_size; i += blockDim.x) {
            if (i + threadIdx.x < B_length - B_consumed) {
                B_S[i + threadIdx.x] = B[B_curr + B_consumed + i + threadIdx.x];
            }
        }
        __syncthreads();

        // Fig. 13.13 Part 3: every thread claims a slice of this tile's C
        // output, co_rank()s it against the SHARED-memory copy, and merges.
        int remainingC = C_length - C_completed;
        int perThread = tile_size / blockDim.x;
        int c_curr = threadIdx.x * perThread;
        c_curr = c_curr < remainingC ? c_curr : remainingC;
        int c_next = (threadIdx.x + 1) * perThread;
        c_next = c_next < remainingC ? c_next : remainingC;

        int tileA_len = A_length - A_consumed;
        tileA_len = tileA_len < tile_size ? tileA_len : tile_size;
        int tileB_len = B_length - B_consumed;
        tileB_len = tileB_len < tile_size ? tileB_len : tile_size;

        int a_curr = co_rank(c_curr, A_S, tileA_len, B_S, tileB_len);
        int b_curr = c_curr - a_curr;
        int a_next = co_rank(c_next, A_S, tileA_len, B_S, tileB_len);
        int b_next = c_next - a_next;

        merge_sequential(A_S + a_curr, a_next - a_curr, B_S + b_curr, b_next - b_curr,
                          C + C_curr + C_completed + c_curr);
        __syncthreads();  // all threads done reading A_S/B_S before next tile overwrites them

        // Book's own bookkeeping (§13.6, end of Fig. 13.13): the book states
        // this as co_rank(tile_size, A_S, tile_size, B_S, tile_size), which
        // is correct whenever both tiles are fully populated (tileA_len ==
        // tileB_len == tile_size, true throughout its own worked example).
        // With a skewed input (e.g. one array much shorter than the
        // other -- exercised by this file's "A empty" test case below), a
        // tile can still be genuinely PARTIAL on a non-final iteration, so
        // the literal tile_size bound would read uninitialized shared
        // memory as if it were valid data and corrupt A_consumed for the
        // NEXT iteration (not harmless in that case). Using the tile's
        // actual valid lengths (tileA_len/tileB_len, already computed
        // above for this same iteration's per-thread co_rank calls)
        // generalizes correctly and is identical to the book's formula
        // whenever the tile genuinely is full.
        C_completed += tile_size;
        A_consumed += co_rank(tileA_len + tileB_len < tile_size ? tileA_len + tileB_len : tile_size,
                               A_S, tileA_len, B_S, tileB_len);
        B_consumed = C_completed - A_consumed;
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Test harness
// ---------------------------------------------------------------------------
std::vector<int> generateSorted(int n, unsigned int seed, int valueRange) {
    std::vector<int> v(n);
    unsigned int state = seed;
    for (int i = 0; i < n; ++i) {
        state = state * 1103515245u + 12345u;
        v[i] = static_cast<int>((state >> 8) % valueRange);
    }
    std::sort(v.begin(), v.end());
    return v;
}

bool runTestCase(int m, int n, int gridDimX, int blockDimX, int tile_size, const char *label) {
    std::vector<int> A_h = generateSorted(m, 555111u, 1 << 20);
    std::vector<int> B_h = generateSorted(n, 222888u, 1 << 20);

    std::vector<int> ref(m + n);
    merge_sequential(A_h.data(), m, B_h.data(), n, ref.data());

    int *A_d, *B_d, *C_d;
    CUDA_CHECK(cudaMalloc(&A_d, (m > 0 ? m : 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&B_d, (n > 0 ? n : 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&C_d, (m + n) * sizeof(int)));
    if (m > 0) CUDA_CHECK(cudaMemcpy(A_d, A_h.data(), m * sizeof(int), cudaMemcpyHostToDevice));
    if (n > 0) CUDA_CHECK(cudaMemcpy(B_d, B_h.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    size_t shmemBytes = 2 * tile_size * sizeof(int);

    // Warm-up (untimed) then timed launch, per this project's fairness rule.
    mergeTiledKernel<<<gridDimX, blockDimX, shmemBytes>>>(A_d, m, B_d, n, C_d, tile_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    mergeTiledKernel<<<gridDimX, blockDimX, shmemBytes>>>(A_d, m, B_d, n, C_d, tile_size);
    CUDA_CHECK(cudaGetLastError());
    float ms = timer.stopAndGetMs();

    std::vector<int> C_h(m + n);
    CUDA_CHECK(cudaMemcpy(C_h.data(), C_d, (m + n) * sizeof(int), cudaMemcpyDeviceToHost));

    bool ok = (C_h == ref);
    printf("%s: m=%d n=%d grid=%d block=%d tile_size=%d: %.4f ms  [%s]\n", label, m, n, gridDimX,
           blockDimX, tile_size, ms, ok ? "match" : "MISMATCH");

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
    return ok;
}

int main() {
    printf("Tiled merge kernel (§13.6, Figs. 13.10-13.13):\n");
    bool ok = true;
    // Book's own worked example verbatim: 33000+31000=64000 elements,
    // 16 blocks * 4000 elements/block, tile_size=1024 -> 4 iterations/block
    // with a partial (4000 - 3*1024 = 928-element) last tile.
    ok = runTestCase(33000, 31000, 16, 128, 1024, "book example") && ok;
    // Larger, more blocks, smaller tile relative to per-block work ->
    // several more full+partial iterations, different m/n ratio.
    ok = runTestCase(500000, 380000, 64, 128, 512, "larger/random") && ok;
    // Degenerate sizes.
    ok = runTestCase(1, 1, 1, 128, 1024, "tiny") && ok;
    ok = runTestCase(0, 4000, 4, 128, 512, "A empty") && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
