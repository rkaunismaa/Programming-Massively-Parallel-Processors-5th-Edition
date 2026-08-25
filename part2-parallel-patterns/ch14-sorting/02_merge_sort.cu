// Chapter 14: Sorting
// §14.3, Fig. 14.3: parallel merge sort.
//
// Merge sort divides the input into many independently-sorted segments,
// then repeatedly merges pairs of adjacent sorted segments into
// double-length segments until only one segment (the fully sorted array)
// remains. §14.3's text describes this at the algorithm level and points
// at the parallel merge pattern of Chapter 13 to do the merging ("We have
// already seen how to parallelize a merge operation in Chapter 13"), but
// explicitly leaves assembling the two into a full parallel merge sort to
// the reader. Per this chapter's task brief, this file builds that parallel
// merge sort with a merge kernel implemented LOCALLY here (co_rank() and a
// co-rank-based merge kernel, in the same style as
// ch13-merge/01_merge_basic_parallel.cu) rather than including Ch13's
// files -- every chapter in this project is self-contained.
//
// Stage structure (Fig. 14.3): starting from segments of length
// INITIAL_SEG_LEN (each sorted independently, in parallel, one thread per
// segment), every stage doubles the segment length by merging each pair of
// adjacent same-length sorted segments into one double-length sorted
// segment, until segLen >= n. The number of stages is O(log(n /
// INITIAL_SEG_LEN)) = O(log n), matching §14.3's stated O(log N) iteration
// count. To keep every stage's segment pairing exact (no partial last
// segment to special-case), n is required to be an exact power-of-two
// multiple of the initial segment length in every test case below.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#include "../../common/cuda_utils.h"

#define BLOCK_DIM 256

// ---------------------------------------------------------------------------
// Co-rank function (as in Ch. 13, §13.4, Fig. 13.5) -- reimplemented locally
// per this chapter's self-containment rule. Returns i, the co-rank of k in
// A (caller derives j = k - i): the unique split (i, j) with i + j = k such
// that merging A[0..i-1] and B[0..j-1] produces exactly the first k
// elements of the stable merge of A and B.
// ---------------------------------------------------------------------------
__device__ int co_rank(int k, const int *A, int m, const int *B, int n) {
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

// ---------------------------------------------------------------------------
// Initial segment sort: one thread sorts one length-segLen segment in place
// with a sequential insertion sort. This is the "sorting each segment
// independently" step in Fig. 14.3 -- the book leaves the choice of
// per-segment sort algorithm open; insertion sort is simple and correct for
// the small segment lengths used here.
// ---------------------------------------------------------------------------
__global__ void initialSegmentSortKernel(int *data, int n, int segLen) {
    int seg = blockIdx.x * blockDim.x + threadIdx.x;
    int start = seg * segLen;
    if (start >= n) return;
    int end = start + segLen;  // n is always an exact multiple of segLen here
    for (int i = start + 1; i < end; ++i) {
        int key = data[i];
        int j = i - 1;
        while (j >= start && data[j] > key) {
            data[j + 1] = data[j];
            --j;
        }
        data[j + 1] = key;
    }
}

// ---------------------------------------------------------------------------
// §14.3, Fig. 14.3 merge stage, built on the Ch. 13 co-rank pattern: one
// thread per OUTPUT element. Each pair of adjacent length-segLen sorted
// segments [pairStart, pairStart+segLen) and [pairStart+segLen,
// pairStart+2*segLen) is merged independently; a thread's global index idx
// determines which pair it belongs to (idx / (2*segLen)) and its local rank
// k within that pair's output. co_rank(k) gives the exact split (i, j) in
// the two input segments, and the element this thread writes is simply
// whichever of A[i]/B[j] is next in stable merge order at that split point.
// ---------------------------------------------------------------------------
__global__ void mergeStageKernel(const int *in, int *out, int n, int segLen) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int pairSize = 2 * segLen;
    int pairStart = (idx / pairSize) * pairSize;
    int k = idx - pairStart;

    const int *A = in + pairStart;
    const int *B = in + pairStart + segLen;

    int i = co_rank(k, A, segLen, B, segLen);
    int j = k - i;

    int val;
    if (i < segLen && (j >= segLen || A[i] <= B[j])) {
        val = A[i];
    } else {
        val = B[j];
    }
    out[idx] = val;
}

std::vector<int> generateInput(int n, unsigned int seed, int valueRange) {
    std::vector<int> v(n);
    unsigned int state = seed;
    for (int i = 0; i < n; ++i) {
        state = state * 1103515245u + 12345u;
        v[i] = static_cast<int>((state >> 8) % valueRange);
    }
    return v;
}

float runMergeSort(const std::vector<int> &input_h, std::vector<int> &out_h, int segLen0) {
    int n = static_cast<int>(input_h.size());
    size_t bytes = n * sizeof(int);

    int *bufA_d, *bufB_d;
    CUDA_CHECK(cudaMalloc(&bufA_d, bytes));
    CUDA_CHECK(cudaMalloc(&bufB_d, bytes));

    dim3 block(BLOCK_DIM);
    dim3 gridMerge((n + BLOCK_DIM - 1) / BLOCK_DIM);
    int numSegs = n / segLen0;
    dim3 gridSeg((numSegs + BLOCK_DIM - 1) / BLOCK_DIM);

    auto resetInput = [&]() {
        CUDA_CHECK(cudaMemcpy(bufA_d, input_h.data(), bytes, cudaMemcpyHostToDevice));
    };

    auto sortPass = [&]() -> int * {
        initialSegmentSortKernel<<<gridSeg, block>>>(bufA_d, n, segLen0);
        CUDA_CHECK(cudaGetLastError());

        int *cur = bufA_d, *nxt = bufB_d;
        for (int segLen = segLen0; segLen < n; segLen *= 2) {
            mergeStageKernel<<<gridMerge, block>>>(cur, nxt, n, segLen);
            CUDA_CHECK(cudaGetLastError());
            std::swap(cur, nxt);
        }
        return cur;
    };

    // Warm-up pass (untimed).
    resetInput();
    sortPass();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed pass on freshly reset input.
    resetInput();
    GpuTimer timer;
    timer.start();
    int *result_d = sortPass();
    float ms = timer.stopAndGetMs();

    out_h.resize(n);
    CUDA_CHECK(cudaMemcpy(out_h.data(), result_d, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(bufA_d));
    CUDA_CHECK(cudaFree(bufB_d));
    return ms;
}

bool runTestCase(int n, int segLen0) {
    std::vector<int> input_h = generateInput(n, 987u + static_cast<unsigned int>(n) + segLen0, 1 << 20);

    std::vector<int> ref = input_h;
    std::sort(ref.begin(), ref.end());

    std::vector<int> gpu_h;
    float ms = runMergeSort(input_h, gpu_h, segLen0);

    bool ok = (gpu_h == ref);
    int numStages = 0;
    for (int segLen = segLen0; segLen < n; segLen *= 2) ++numStages;
    printf("n=%d segLen0=%d (segments=%d, merge stages=%d): %.4f ms  [%s]\n", n, segLen0, n / segLen0,
           numStages, ms, ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    printf("Parallel merge sort (§14.3, Fig. 14.3):\n");
    bool ok = true;
    ok = runTestCase(256, 16) && ok;
    ok = runTestCase(1 << 16, 64) && ok;
    ok = runTestCase(4096, 1) && ok;  // segLen0=1: no initial-sort work, pure merge-doubling
    ok = runTestCase(1, 1) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
