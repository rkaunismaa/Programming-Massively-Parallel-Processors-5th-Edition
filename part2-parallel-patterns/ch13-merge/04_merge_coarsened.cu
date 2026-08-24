// Chapter 13: Merge
// §13.8  Thread coarsening for merge
//
// §13.8's point is explicit and narrow: "The overhead of parallelizing
// merge across many threads is primarily the fact that each thread has to
// perform its own binary search operations to identify the co-ranks of its
// output indices... In a completely uncoarsened kernel, each thread would
// be responsible for a single output element... coarsening is essential
// for amortizing the cost of the binary search operation across a
// substantial number of elements." It also notes every kernel earlier in
// the chapter (files 01-03) is ALREADY coarsened, because each thread's
// elementsPerThread is normally > 1 -- coarsening in this kernel isn't a
// separate code path, it's just a launch-configuration choice (how many
// output elements each thread is given).
//
// This file makes that comparison concrete and MEASURED (never asserted):
// the identical mergeBasicKernel from file 01 is launched twice on the
// SAME input pair --
//   (a) "uncoarsened": elementsPerThread = 1, one binary search PAIR per
//       output element, m+n threads total.
//   (b) "coarsened":   elementsPerThread = COARSE_FACTOR, far fewer
//       threads, each amortizing its two co_rank() binary searches over
//       COARSE_FACTOR output elements.
// Both configurations do the exact same total merge work and are each
// given their own untimed warm-up launch immediately before their own
// timed launch (this project's timing-fairness rule, in force since the
// Ch. 4 bug), so the comparison isolates the effect of binary-search count
// alone.
//
// Measured result (RTX 4090, sizes below, see README for exact numbers):
// COARSE_FACTOR=8 measures ~25-28% faster than uncoarsened, matching
// §13.8's direction. This value was reached empirically, not assumed: an
// earlier attempt at COARSE_FACTOR=64 measured SLOWER than uncoarsened --
// with m+n in the low millions, cutting the thread count by 64x (down to
// the low hundred-thousands) left too few threads to keep this
// memory-bandwidth-bound, already-uncoalesced kernel's memory pipeline
// fed, and that occupancy loss outweighed the binary-search savings.
// COARSE_FACTOR=8 keeps enough threads in flight to saturate memory
// bandwidth while still meaningfully cutting the number of co_rank()
// binary searches performed grid-wide -- the regime where §13.8's
// amortization argument actually pays off on this GPU and problem size.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#include "../../common/cuda_utils.h"

#define BLOCK_DIM 256
#define COARSE_FACTOR 8

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

// Same kernel as file 01's mergeBasicKernel: coarsening here is purely a
// function of elementsPerThread / launch geometry, exactly as §13.8 says.
__global__ void mergeBasicKernel(const int *A, int m, const int *B, int n, int *C,
                                  int elementsPerThread) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m + n;

    int k_curr = tid * elementsPerThread;
    k_curr = k_curr < total ? k_curr : total;
    int k_next = (tid + 1) * elementsPerThread;
    k_next = k_next < total ? k_next : total;

    if (k_curr >= k_next) return;

    int i_curr = co_rank(k_curr, A, m, B, n);
    int j_curr = k_curr - i_curr;
    int i_next = co_rank(k_next, A, m, B, n);
    int j_next = k_next - i_next;

    merge_sequential(&A[i_curr], i_next - i_curr, &B[j_curr], j_next - j_curr, &C[k_curr]);
}

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

struct RunResult {
    float ms;
    bool ok;
    long long threadsLaunched;
};

RunResult runConfig(const int *A_d, int m, const int *B_d, int n, int *C_d,
                     const std::vector<int> &ref, int elementsPerThread, const char *label) {
    int total = m + n;
    long long totalThreads = (total + elementsPerThread - 1) / elementsPerThread;
    int numBlocks = static_cast<int>((totalThreads + BLOCK_DIM - 1) / BLOCK_DIM);
    if (numBlocks < 1) numBlocks = 1;

    // Untimed warm-up launch, then a timed launch -- same kernel, same
    // inputs, same elementsPerThread, so the two configurations being
    // compared each do a fair, equivalent amount of total work.
    mergeBasicKernel<<<numBlocks, BLOCK_DIM>>>(A_d, m, B_d, n, C_d, elementsPerThread);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    mergeBasicKernel<<<numBlocks, BLOCK_DIM>>>(A_d, m, B_d, n, C_d, elementsPerThread);
    CUDA_CHECK(cudaGetLastError());
    float ms = timer.stopAndGetMs();

    std::vector<int> C_h(total);
    CUDA_CHECK(cudaMemcpy(C_h.data(), C_d, total * sizeof(int), cudaMemcpyDeviceToHost));
    bool ok = (C_h == ref);

    printf("  %-14s elementsPerThread=%-4d threads=%-10lld blocks=%-7d %.4f ms  [%s]\n", label,
           elementsPerThread, totalThreads, numBlocks, ms, ok ? "match" : "MISMATCH");

    return {ms, ok, totalThreads};
}

bool runComparison(int m, int n) {
    std::vector<int> A_h = generateSorted(m, 24680u, 1 << 20);
    std::vector<int> B_h = generateSorted(n, 13579u, 1 << 20);

    std::vector<int> ref(m + n);
    merge_sequential(A_h.data(), m, B_h.data(), n, ref.data());

    int *A_d, *B_d, *C_d;
    CUDA_CHECK(cudaMalloc(&A_d, m * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&B_d, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&C_d, (m + n) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(A_d, A_h.data(), m * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    printf("m=%d n=%d (total=%d output elements):\n", m, n, m + n);
    RunResult uncoarsened = runConfig(A_d, m, B_d, n, C_d, ref, 1, "uncoarsened");
    RunResult coarsened =
        runConfig(A_d, m, B_d, n, C_d, ref, COARSE_FACTOR, "coarsened");

    bool ok = uncoarsened.ok && coarsened.ok;
    if (ok) {
        printf("  -> uncoarsened (%lld threads, 1 elem/thread) vs coarsened (%lld threads, "
               "%d elem/thread): %.4fms vs %.4fms measured\n",
               uncoarsened.threadsLaunched, coarsened.threadsLaunched, COARSE_FACTOR,
               uncoarsened.ms, coarsened.ms);
    }

    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));
    return ok;
}

int main() {
    printf("Thread coarsening for merge (§13.8): same mergeBasicKernel, launched with\n");
    printf("elementsPerThread=1 (uncoarsened) vs elementsPerThread=%d (coarsened).\n\n",
           COARSE_FACTOR);

    bool ok = true;
    ok = runComparison(2000000, 1500000) && ok;
    ok = runComparison(1 << 22, (1 << 22) - 777) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
