// Chapter 14: Sorting
// §14.2, Figs. 14.1-14.2: parallel odd-even (transposition) sort.
//
// Odd-even sort alternates between two disjoint sets of adjacent-element
// comparisons so that a whole "sweep" can run as one race-free parallel
// kernel launch: an EVEN step compares/swaps pairs (0,1), (2,3), (4,5), ...
// and an ODD step compares/swaps pairs (1,2), (3,4), (5,6), .... Within one
// step every pair is disjoint from every other pair, so threads never
// contend for the same array slot. The host alternates even/odd steps,
// launching one kernel per step, until a full even+odd round produces no
// swaps (the list is sorted) -- capped at N steps, the bound the book gives
// for guaranteed convergence (§14.2: "the number of iterations needed is
// O(N)... in the worst case, the largest element is at the beginning of the
// list and needs N iterations to reach the end").
//
// §14.2's text (discussing Fig. 14.2, line 10) calls out that writing 1 to
// a shared `hasChanged` flag from many threads is technically a data race,
// but a BENIGN one because every writer stores the identical value
// (idempotence) -- while noting this still violates the C++ memory model in
// principle and that "conservative programmers should use an atomic
// operation." This file takes that advice and uses atomicExch.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#include "../../common/cuda_utils.h"

#define BLOCK_DIM 256

// ---------------------------------------------------------------------------
// §14.2, Fig. 14.2: one odd-even sort step. Launched with ceil(n/2) threads;
// thread t is responsible for the pair starting at index 2*t (even step) or
// 2*t+1 (odd step, isOddStep=1).
// ---------------------------------------------------------------------------
__global__ void oddEvenSortKernel(int *data, int n, int isOddStep, int *hasChanged) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = 2 * tid + isOddStep;
    if (i + 1 < n) {
        if (data[i] > data[i + 1]) {
            int tmp = data[i];
            data[i] = data[i + 1];
            data[i + 1] = tmp;
            atomicExch(hasChanged, 1);
        }
    }
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

// ---------------------------------------------------------------------------
// Runs the full even/odd sweep loop to convergence (or the N-step cap) on
// whatever is currently in data_d, in place.
// ---------------------------------------------------------------------------
void oddEvenSortInPlace(int *data_d, int n, int *hasChanged_d, dim3 grid, dim3 block) {
    int prevChanged = 1;
    for (int step = 0; step < n; ++step) {
        int isOddStep = step & 1;
        int zero = 0;
        CUDA_CHECK(cudaMemcpy(hasChanged_d, &zero, sizeof(int), cudaMemcpyHostToDevice));
        oddEvenSortKernel<<<grid, block>>>(data_d, n, isOddStep, hasChanged_d);
        CUDA_CHECK(cudaGetLastError());
        int changed = 0;
        CUDA_CHECK(cudaMemcpy(&changed, hasChanged_d, sizeof(int), cudaMemcpyDeviceToHost));
        if (changed == 0 && prevChanged == 0) break;  // a full even+odd round made no swaps
        prevChanged = changed;
    }
}

float runOddEvenSort(const std::vector<int> &input_h, std::vector<int> &out_h) {
    int n = static_cast<int>(input_h.size());
    size_t bytes = n * sizeof(int);

    int *data_d, *hasChanged_d;
    CUDA_CHECK(cudaMalloc(&data_d, bytes));
    CUDA_CHECK(cudaMalloc(&hasChanged_d, sizeof(int)));

    int threads = (n + 1) / 2;
    dim3 block(BLOCK_DIM);
    dim3 grid((threads + BLOCK_DIM - 1) / BLOCK_DIM);

    // Warm-up pass (untimed) on a throw-away copy of the unsorted input --
    // this project's timing-fairness convention, applied at the whole
    // multi-launch-loop level since a single kernel launch isn't the unit
    // of work here.
    CUDA_CHECK(cudaMemcpy(data_d, input_h.data(), bytes, cudaMemcpyHostToDevice));
    oddEvenSortInPlace(data_d, n, hasChanged_d, grid, block);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed pass on fresh unsorted data.
    CUDA_CHECK(cudaMemcpy(data_d, input_h.data(), bytes, cudaMemcpyHostToDevice));
    GpuTimer timer;
    timer.start();
    oddEvenSortInPlace(data_d, n, hasChanged_d, grid, block);
    float ms = timer.stopAndGetMs();

    out_h.resize(n);
    CUDA_CHECK(cudaMemcpy(out_h.data(), data_d, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(data_d));
    CUDA_CHECK(cudaFree(hasChanged_d));
    return ms;
}

bool runTestCase(int n) {
    std::vector<int> input_h = generateInput(n, 12345u + n, 1 << 20);

    std::vector<int> ref = input_h;
    std::sort(ref.begin(), ref.end());

    std::vector<int> gpu_h;
    float ms = runOddEvenSort(input_h, gpu_h);

    bool ok = (gpu_h == ref);
    printf("n=%d: %.4f ms  [%s]\n", n, ms, ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    printf("Parallel odd-even sort (§14.2, Fig. 14.2):\n");
    bool ok = true;
    ok = runTestCase(1) && ok;
    ok = runTestCase(2) && ok;
    ok = runTestCase(1000) && ok;
    ok = runTestCase(2047) && ok;   // odd n
    ok = runTestCase(4096) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
