// Chapter 13: Merge
// §13.7  A circular-buffer merge kernel (Figs. 13.15-13.20)
//
// File 02's tiled kernel wastes half its shared-memory bandwidth: every
// while-loop iteration reloads a FULL tile_size elements of A and B,
// discarding whatever fraction of the previous tile went unused (in the
// worst case, a whole tile could still hold elements the block hasn't
// consumed yet). §13.7's fix is a circular buffer: A_S/B_S keep whatever
// unconsumed elements remain from the previous iteration in place, and
// each new iteration only tops up the buffer with exactly as many fresh
// elements as were consumed (A_S_consumed / B_S_consumed), wrapping the
// write position around the end of the tile_size-sized array with modulo
// arithmetic (A_S_start / B_S_start track where each buffer's logical
// "start" currently sits).
//
// Per §13.7's own "simplified model" (Fig. 13.17b): co_rank_circular() and
// merge_sequential_circular() take the SAME i/j bookkeeping as the plain
// co_rank()/merge_sequential(), and only translate an offset into an
// actual circular-buffer index -- via (A_S_start + offset) % tile_size --
// at the point where they touch A_S/B_S. This keeps the search/merge logic
// itself untouched from files 01-02; only element ACCESS changes.
//
// Book-fidelity note on A_S_consumed/B_S_consumed initialization: the text
// states index arithmetic for the load loop as
// "A_S_start+(tile_size-A_S_consumed)+i+threadIdx" (quoted verbatim,
// §13.7), which only produces a full-tile initial load (as Fig. 13.15(a)
// shows for iteration 0, when A_S_start=0) if A_S_consumed=tile_size going
// into the FIRST iteration. That is the value used below, together with
// A_S_start=B_S_start=0, so iteration 0 loads a full fresh tile exactly as
// Fig. 13.15(a) depicts.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#include "../../common/cuda_utils.h"

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

// ---------------------------------------------------------------------------
// §13.7, Fig. 13.19: co-rank over a circular buffer. Identical i/j/i_low/
// j_low bookkeeping to co_rank(); only the array-access index is remapped
// through the circular offset (A_S_start/B_S_start, wrapped mod tile_size).
// The A_S[i-1]/B_S[j] access is only ever reached when i>0 / j>0 (guarded
// by the enclosing && before the wrap is computed), so the "-1" offset is
// always >= 0 before the modulo -- no negative-modulo case ever arises.
// ---------------------------------------------------------------------------
__host__ __device__ int co_rank_circular(int k, const int *A_S, int m, const int *B_S, int n,
                                          int A_S_start, int B_S_start, int tile_size) {
    int i = k < m ? k : m;
    int j = k - i;
    int i_low = 0 > (k - n) ? 0 : (k - n);
    int j_low = 0 > (k - m) ? 0 : (k - m);
    int delta;
    bool active = true;
    while (active) {
        bool iTooHigh = false;
        if (i > 0 && j < n) {
            int i_m_1_cir = (A_S_start + i - 1) % tile_size;
            int j_cir = (B_S_start + j) % tile_size;
            iTooHigh = A_S[i_m_1_cir] > B_S[j_cir];
        }
        if (iTooHigh) {
            delta = (i - i_low + 1) >> 1;
            j_low = j;
            j += delta;
            i -= delta;
            continue;
        }
        bool jTooHigh = false;
        if (j > 0 && i < m) {
            int j_m_1_cir = (B_S_start + j - 1) % tile_size;
            int i_cir = (A_S_start + i) % tile_size;
            jTooHigh = B_S[j_m_1_cir] >= A_S[i_cir];
        }
        if (jTooHigh) {
            delta = (j - j_low + 1) >> 1;
            i_low = i;
            i += delta;
            j -= delta;
            continue;
        }
        active = false;
    }
    return i;
}

// §13.7, Fig. 13.20: sequential merge over a circular buffer -- same logic
// as merge_sequential(), with every A/B access remapped through the
// circular offset.
__host__ __device__ void merge_sequential_circular(const int *A_S, int m, const int *B_S, int n,
                                                     int *C, int A_S_start, int B_S_start,
                                                     int tile_size) {
    int i = 0, j = 0, k = 0;
    while (i < m && j < n) {
        int i_cir = (A_S_start + i) % tile_size;
        int j_cir = (B_S_start + j) % tile_size;
        if (A_S[i_cir] <= B_S[j_cir]) {
            C[k++] = A_S[i_cir];
            ++i;
        } else {
            C[k++] = B_S[j_cir];
            ++j;
        }
    }
    while (i < m) {
        int i_cir = (A_S_start + i) % tile_size;
        C[k++] = A_S[i_cir];
        ++i;
    }
    while (j < n) {
        int j_cir = (B_S_start + j) % tile_size;
        C[k++] = B_S[j_cir];
        ++j;
    }
}

// ---------------------------------------------------------------------------
// §13.7: circular-buffer merge kernel. Part 1 (block-level co-rank) is
// identical to file 02's; Parts 2-3 use the circular buffer.
// ---------------------------------------------------------------------------
__global__ void mergeCircularBufferKernel(const int *A, int m, const int *B, int n, int *C,
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

    __shared__ int blockCorank[2];
    if (threadIdx.x == 0) {
        blockCorank[0] = co_rank(C_curr, A, m, B, n);
        blockCorank[1] = co_rank(C_next, A, m, B, n);
    }
    __syncthreads();
    int A_curr = blockCorank[0];
    int A_next = blockCorank[1];
    int B_curr = C_curr - A_curr;
    int B_next = C_next - A_next;
    __syncthreads();

    int C_length = C_next - C_curr;
    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;
    int total_iteration = (C_length + tile_size - 1) / tile_size;
    int C_completed = 0;

    int A_S_start = 0;
    int B_S_start = 0;
    // A_loaded/B_loaded: TOTAL elements ever fetched from global A/B so far
    // (monotonically increasing). leftoverA/leftoverB: elements currently
    // resident in the circular buffer that have NOT yet been merged (carried
    // over, unread, from the previous iteration).
    //
    // Book-fidelity note: §13.7's text gives the refill destination index as
    // "A_S_start+(tile_size-A_S_consumed)+i+threadIdx", which is only valid
    // when the PREVIOUS tile was fully populated (tile_size elements) before
    // this iteration's consumption -- true throughout the book's own worked
    // example (A and B both always have >= tile_size elements remaining
    // until each block's very last iteration). Reusing "elements consumed by
    // the merge" as the refill request breaks once loaded != consumed+
    // leftover in general (this file's larger/skewed test cases below hit
    // that in intermediate iterations, not just the final one). Tracking
    // A_loaded/leftoverA explicitly and requesting exactly
    // (tile_size - leftoverA) new elements (capped by what remains in
    // global A) generalizes correctly while reducing to the book's formula
    // exactly whenever the tile is genuinely full.
    long long A_loaded = 0, B_loaded = 0;
    int leftoverA = 0, leftoverB = 0;

    for (int counter = 0; counter < total_iteration; ++counter) {
        int reqA = tile_size - leftoverA;
        int availA = static_cast<int>(A_length - A_loaded);
        reqA = reqA < availA ? reqA : availA;
        int reqB = tile_size - leftoverB;
        int availB = static_cast<int>(B_length - B_loaded);
        reqB = reqB < availB ? reqB : availB;

        // Fig. 13.16: top up exactly the freed slots, writing right after
        // the surviving leftover region, wrapped modulo tile_size.
        for (int i = 0; i < reqA; i += blockDim.x) {
            int idx = i + threadIdx.x;
            if (idx < reqA) {
                int dst = (A_S_start + leftoverA + idx) % tile_size;
                A_S[dst] = A[A_curr + A_loaded + idx];
            }
        }
        for (int i = 0; i < reqB; i += blockDim.x) {
            int idx = i + threadIdx.x;
            if (idx < reqB) {
                int dst = (B_S_start + leftoverB + idx) % tile_size;
                B_S[dst] = B[B_curr + B_loaded + idx];
            }
        }
        __syncthreads();

        A_loaded += reqA;
        B_loaded += reqB;
        int tileA_len = leftoverA + reqA;
        int tileB_len = leftoverB + reqB;

        int remainingC = C_length - C_completed;
        int tileC = remainingC < tile_size ? remainingC : tile_size;
        int perThread = tile_size / blockDim.x;
        int c_curr = threadIdx.x * perThread;
        c_curr = c_curr < tileC ? c_curr : tileC;
        int c_next = (threadIdx.x + 1) * perThread;
        c_next = c_next < tileC ? c_next : tileC;

        int a_curr = co_rank_circular(c_curr, A_S, tileA_len, B_S, tileB_len, A_S_start,
                                       B_S_start, tile_size);
        int b_curr = c_curr - a_curr;
        int a_next = co_rank_circular(c_next, A_S, tileA_len, B_S, tileB_len, A_S_start,
                                       B_S_start, tile_size);
        int b_next = c_next - a_next;

        merge_sequential_circular(A_S, a_next - a_curr, B_S, b_next - b_curr,
                                   C + C_curr + C_completed + c_curr, A_S_start + a_curr,
                                   B_S_start + b_curr, tile_size);
        __syncthreads();

        // End-of-iteration bookkeeping: how much of THIS tile did the block
        // actually consume? On the final iteration this may run over a
        // partially-filled tile (harmless -- see file 02's analogous note).
        int A_S_consumed = co_rank_circular(tileC, A_S, tileA_len, B_S, tileB_len, A_S_start,
                                             B_S_start, tile_size);
        int B_S_consumed = tileC - A_S_consumed;
        leftoverA = tileA_len - A_S_consumed;
        leftoverB = tileB_len - B_S_consumed;
        C_completed += tileC;

        A_S_start = (A_S_start + A_S_consumed) % tile_size;
        B_S_start = (B_S_start + B_S_consumed) % tile_size;
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Test harness (same style as file 02, so results are directly comparable).
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

// CPU reference merge (plain, non-circular) -- identical semantics.
static void merge_sequential_ref(const int *A, int m, const int *B, int n, int *C) {
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

bool runTestCase(int m, int n, int gridDimX, int blockDimX, int tile_size, const char *label) {
    std::vector<int> A_h = generateSorted(m, 111333u, 1 << 20);
    std::vector<int> B_h = generateSorted(n, 444777u, 1 << 20);

    std::vector<int> ref(m + n);
    merge_sequential_ref(A_h.data(), m, B_h.data(), n, ref.data());

    int *A_d, *B_d, *C_d;
    CUDA_CHECK(cudaMalloc(&A_d, (m > 0 ? m : 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&B_d, (n > 0 ? n : 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&C_d, (m + n) * sizeof(int)));
    if (m > 0) CUDA_CHECK(cudaMemcpy(A_d, A_h.data(), m * sizeof(int), cudaMemcpyHostToDevice));
    if (n > 0) CUDA_CHECK(cudaMemcpy(B_d, B_h.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    size_t shmemBytes = 2 * tile_size * sizeof(int);

    mergeCircularBufferKernel<<<gridDimX, blockDimX, shmemBytes>>>(A_d, m, B_d, n, C_d, tile_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    mergeCircularBufferKernel<<<gridDimX, blockDimX, shmemBytes>>>(A_d, m, B_d, n, C_d, tile_size);
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
    printf("Circular-buffer tiled merge kernel (§13.7):\n");
    bool ok = true;
    ok = runTestCase(33000, 31000, 16, 128, 1024, "book example") && ok;
    ok = runTestCase(500000, 380000, 64, 128, 512, "larger/random") && ok;
    ok = runTestCase(1, 1, 1, 128, 1024, "tiny") && ok;
    ok = runTestCase(0, 4000, 4, 128, 512, "A empty") && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
