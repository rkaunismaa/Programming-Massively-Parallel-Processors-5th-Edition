// Chapter 13: Merge
// §13.2  A sequential merge algorithm (Fig. 13.2)
// §13.4  Co-rank function implementation (Fig. 13.5)
// §13.5  A basic parallel merge kernel (Fig. 13.9)
//
// The merge pattern combines two sorted arrays A (m elements) and B (n
// elements) into one sorted array C (m+n elements). §13.1 defines
// STABILITY: whenever A and B have equal-valued elements, the A element
// must appear first in C (this is the tie-breaking rule this whole chapter,
// and every file in this directory, uses -- see merge_sequential below).
//
// The key idea (§13.3, Observation 2) is the CO-RANK function: for any
// output rank k (0 <= k <= m+n), there is a unique pair (i, j) with
// i + j = k such that C[0..k-1] is exactly the merge of A[0..i-1] and
// B[0..j-1]. Each thread computes the co-rank of the first and one-past-
// last element of ITS output slice; the difference between consecutive
// co-ranks gives it the exact input sub-arrays to merge sequentially --
// no thread needs to look at any other thread's work.
//
// co_rank() is written __host__ __device__ so the identical implementation
// backs both the GPU kernel (Fig. 13.9) and can be unit-tested directly
// from host code against the book's own worked example.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#include "../../common/cuda_utils.h"

#define BLOCK_DIM 256

// ---------------------------------------------------------------------------
// §13.4, Fig. 13.5: co-rank function (binary search).
//
// Returns i, the co-rank of k in A (the caller derives j = k - i).
// Invariant maintained throughout: i + j == k.
//
// Exit condition (searched for by the while-loop): A[i-1] <= B[j] AND
// B[j-1] < A[i]. Note the asymmetry -- "<=" on the A side, strict "<" on
// the B side. This is exactly what stability requires: on a tie between
// the last element taken from A's prefix and the first element of B's
// suffix, A must have gone first, so A[i-1] == B[j] is an ACCEPTABLE split
// point, but B[j-1] == A[i] is NOT (that would mean a B element was placed
// in C before an equal-valued A element that comes later in this split).
// ---------------------------------------------------------------------------
__host__ __device__ int co_rank(int k, const int *A, int m, const int *B, int n) {
    int i = k < m ? k : m;       // i = min(k, m)
    int j = k - i;
    int i_low = 0 > (k - n) ? 0 : (k - n);  // i_low = max(0, k - n)
    int j_low = 0 > (k - m) ? 0 : (k - m);  // j_low = max(0, k - m)
    int delta;
    bool active = true;
    while (active) {
        if (i > 0 && j < n && A[i - 1] > B[j]) {
            // i is too high: A[i-1] should not be > B[j]. Shrink toward i_low.
            delta = (i - i_low + 1) >> 1;
            j_low = j;
            j += delta;
            i -= delta;
        } else if (j > 0 && i < m && B[j - 1] >= A[i]) {
            // j is too high: B[j-1] must be strictly < A[i]. Shrink toward j_low.
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

// ---------------------------------------------------------------------------
// §13.2, Fig. 13.2: sequential merge of A[0..m-1] and B[0..n-1] into C.
// Stable: on a tie, A's element is written first (A[i] <= B[j] test, not <).
// ---------------------------------------------------------------------------
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
// §13.5, Fig. 13.9: basic parallel merge kernel.
//
// Each thread is assigned a contiguous slice [k_curr, k_next) of the output
// C array (elementsPerThread wide, except possibly the last thread). Two
// co_rank() calls -- one for k_curr, one for k_next -- turn that output
// slice into the exact input sub-arrays this thread must sequentially
// merge. No thread touches another thread's input or output range.
// ---------------------------------------------------------------------------
__global__ void mergeBasicKernel(const int *A, int m, const int *B, int n, int *C,
                                  int elementsPerThread) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m + n;

    int k_curr = tid * elementsPerThread;
    k_curr = k_curr < total ? k_curr : total;
    int k_next = (tid + 1) * elementsPerThread;
    k_next = k_next < total ? k_next : total;

    if (k_curr >= k_next) return;  // this thread has no work

    int i_curr = co_rank(k_curr, A, m, B, n);
    int j_curr = k_curr - i_curr;
    int i_next = co_rank(k_next, A, m, B, n);
    int j_next = k_next - i_next;

    merge_sequential(&A[i_curr], i_next - i_curr, &B[j_curr], j_next - j_curr, &C[k_curr]);
}

// ---------------------------------------------------------------------------
// Unit test: the book's own running example, used throughout §13.3-13.7 and
// restated as Exercise 1: A = {1,7,8,9,10} (m=5), B = {7,10,10,12} (n=4).
// Verifies co_rank() against every worked value the chapter text gives:
//   co_rank(3, A,5, B,4) == 2   (Fig. 13.6-13.8, thread 1's k_curr)
//   co_rank(4, A,5, B,4) == 3   (Fig. 13.4, thread 1's k_curr)
//   co_rank(9, A,5, B,4) == 5   (Fig. 13.4, thread 1's k_next -- both
//                                 arrays fully exhausted)
// and the full stable merge against Fig. 13.1's own stated properties:
// duplicate 10s from B stay in order, and the tied 7 from A precedes the
// tied 7 from B.
// ---------------------------------------------------------------------------
bool testCoRankBookExample() {
    int A[5] = {1, 7, 8, 9, 10};
    int B[4] = {7, 10, 10, 12};

    bool ok = true;
    struct Case { int k, expected_i; };
    Case cases[] = {{3, 2}, {4, 3}, {9, 5}};
    for (auto &c : cases) {
        int i = co_rank(c.k, A, 5, B, 4);
        bool pass = (i == c.expected_i);
        printf("  co_rank(%d, A,5, B,4) = %d (expected %d)  [%s]\n", c.k, i, c.expected_i,
               pass ? "match" : "MISMATCH");
        ok = ok && pass;
    }

    int C[9];
    merge_sequential(A, 5, B, 4, C);
    int expected[9] = {1, 7, 7, 8, 9, 10, 10, 10, 12};
    bool mergeOk = std::equal(C, C + 9, expected);
    printf("  merge_sequential(A,5,B,4) = [");
    for (int idx = 0; idx < 9; ++idx) printf("%d%s", C[idx], idx == 8 ? "" : ",");
    printf("]  [%s]\n", mergeOk ? "match" : "MISMATCH");

    return ok && mergeOk;
}

// ---------------------------------------------------------------------------
// Random sorted-array test harness shared by all four files' style.
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

bool runTestCase(int m, int n) {
    std::vector<int> A_h = generateSorted(m, 12345u, 1 << 20);
    std::vector<int> B_h = generateSorted(n, 987654321u, 1 << 20);

    std::vector<int> ref(m + n);
    merge_sequential(A_h.data(), m, B_h.data(), n, ref.data());

    int *A_d, *B_d, *C_d;
    CUDA_CHECK(cudaMalloc(&A_d, m * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&B_d, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&C_d, (m + n) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(A_d, A_h.data(), m * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    int totalThreads = 65536;
    int elementsPerThread = (m + n + totalThreads - 1) / totalThreads;
    int numBlocks = (totalThreads + BLOCK_DIM - 1) / BLOCK_DIM;

    // Warm-up launch (untimed) before the timed launch -- this project's
    // timing-fairness convention (established after the Ch. 4 timing bug).
    mergeBasicKernel<<<numBlocks, BLOCK_DIM>>>(A_d, m, B_d, n, C_d, elementsPerThread);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    mergeBasicKernel<<<numBlocks, BLOCK_DIM>>>(A_d, m, B_d, n, C_d, elementsPerThread);
    CUDA_CHECK(cudaGetLastError());
    float ms = timer.stopAndGetMs();

    std::vector<int> C_h(m + n);
    CUDA_CHECK(cudaMemcpy(C_h.data(), C_d, (m + n) * sizeof(int), cudaMemcpyDeviceToHost));

    bool ok = (C_h == ref);
    printf("m=%d n=%d (threads=%d, elementsPerThread=%d): %.4f ms  [%s]\n", m, n, totalThreads,
           elementsPerThread, ms, ok ? "match" : "MISMATCH");

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
    return ok;
}

int main() {
    printf("Co-rank unit test (book's worked example, A={1,7,8,9,10} B={7,10,10,12}):\n");
    bool ok = testCoRankBookExample();

    printf("\nBasic parallel merge kernel (Fig. 13.9):\n");
    ok = runTestCase(200000, 150000) && ok;
    ok = runTestCase(1 << 20, (1 << 20) - 12345) && ok;
    ok = runTestCase(1, 1) && ok;
    ok = runTestCase(0, 5000) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
