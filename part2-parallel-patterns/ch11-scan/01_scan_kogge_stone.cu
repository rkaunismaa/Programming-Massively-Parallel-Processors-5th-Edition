// Chapter 11: Scan
// §11.2  Parallel scan with the Kogge-Stone algorithm (Fig. 11.2, Fig. 11.3)
//
// An inclusive scan on operator + (prefix sum) takes input [x0..xN-1] and
// returns [x0, x0+x1, x0+x1+x2, ...]. §11.2's naive "one thread per output,
// sequential reduction" approach is O(N^2) work with an O(N) span -- no
// better than the sequential algorithm, and often worse in practice. The
// Kogge-Stone algorithm (adapted from the 1970s fast-adder design) instead
// runs an IN-PLACE evolving array: after k iterations, position i holds the
// sum of up to 2^k input elements ending at i. Iterating stride = 1, 2, 4,
// ... doubling until stride >= blockDim.x takes log2(N) steps, each doing
// (N - stride) additions, for a total work of N*log2(N) - (N-1) additions
// (§11.5's own derivation) -- O(N log N), better than the naive O(N^2) but
// still worse than the sequential algorithm's O(N).
//
// This is a per-BLOCK (per-segment) scan: as the book states, "we will for
// now implement a kernel where each thread block performs a local parallel
// scan on a segment of the input... Later, in Section 11.9, we will discuss
// how to consolidate these scanned segments to produce a globally scanned
// output array." Consolidation across blocks is file 06 of this chapter.
//
// Fig. 11.3's kernel uses TWO __syncthreads() calls per loop iteration:
//   - the first (before the add) enforces the TRUE dependence: a thread
//     must not read buffer_s values that a previous iteration's writers
//     haven't finished writing yet.
//   - the second (after the add, before the write-back) enforces a FALSE
//     dependence: a thread must not overwrite buffer_s[threadIdx.x] before
//     other active threads have read the old value at that position (thread
//     i+stride reads buffer_s[i] in the same iteration that thread i writes
//     it). The book works through a concrete race (thread 4 vs thread 6,
//     iteration 2) showing the corrupted result if this second barrier is
//     dropped.
// File 02 (§11.3) removes the false dependence -- and its barrier -- via
// double-buffering.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#include "../../common/cuda_utils.h"

// ---------------------------------------------------------------------------
// §11.2, Fig. 11.3: single-buffer Kogge-Stone inclusive scan, one block per
// segment, blockDim.x threads for a blockDim.x-element segment (one thread
// per element). Segment size is the (dynamic) shared-memory allocation,
// sized to blockDim.x at launch.
// ---------------------------------------------------------------------------
__global__ void koggeStoneScanKernel(const float *input, float *output, unsigned int N) {
    extern __shared__ float buffer_s[];

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    buffer_s[threadIdx.x] = (i < N) ? input[i] : 0.0f;

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();  // true dependence: wait for previous iteration's writes
        float temp;
        if (threadIdx.x >= stride) {
            temp = buffer_s[threadIdx.x] + buffer_s[threadIdx.x - stride];
        }
        __syncthreads();  // false dependence: wait for all reads before any write
        if (threadIdx.x >= stride) {
            buffer_s[threadIdx.x] = temp;
        }
    }

    if (i < N) {
        output[i] = buffer_s[threadIdx.x];
    }
}

// Sequential reference, Fig. 11.1 -- applied independently PER SEGMENT of
// segSize elements, matching what the per-block kernel above actually
// computes (each block only sees its own segment; grid-wide consolidation
// is Section 11.9 / file 06).
void scanSegmentsCPU(const float *input, float *output, unsigned int N, unsigned int segSize) {
    for (unsigned int base = 0; base < N; base += segSize) {
        unsigned int end = std::min(base + segSize, N);
        float acc = 0.0f;
        for (unsigned int i = base; i < end; ++i) {
            acc += input[i];
            output[i] = acc;
        }
    }
}

std::vector<float> generateInput(unsigned int n) {
    std::vector<float> v(n);
    unsigned int state = 123456789u;
    for (unsigned int i = 0; i < n; ++i) {
        state = state * 1103515245u + 12345u;
        v[i] = static_cast<float>((state >> 8) & 0xFFFFFF) / static_cast<float>(0x1000000);
    }
    return v;
}

float runKoggeStoneScan(const std::vector<float> &input_h, std::vector<float> &output_h,
                         unsigned int segSize) {
    unsigned int n = static_cast<unsigned int>(input_h.size());
    size_t bytes = n * sizeof(float);
    size_t shmemBytes = segSize * sizeof(float);

    float *input_d, *output_d;
    CUDA_CHECK(cudaMalloc((void **)&input_d, bytes));
    CUDA_CHECK(cudaMalloc((void **)&output_d, bytes));
    CUDA_CHECK(cudaMemcpy(input_d, input_h.data(), bytes, cudaMemcpyHostToDevice));

    dim3 dimBlock(segSize);
    dim3 dimGrid(n / segSize);

    koggeStoneScanKernel<<<dimGrid, dimBlock, shmemBytes>>>(input_d, output_d, n);  // warm-up
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    koggeStoneScanKernel<<<dimGrid, dimBlock, shmemBytes>>>(input_d, output_d, n);
    CUDA_CHECK(cudaGetLastError());
    float ms = timer.stopAndGetMs();

    output_h.resize(n);
    CUDA_CHECK(cudaMemcpy(output_h.data(), output_d, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(input_d));
    CUDA_CHECK(cudaFree(output_d));
    return ms;
}

bool runTestCase(unsigned int segSize, unsigned int numBlocks) {
    unsigned int n = segSize * numBlocks;
    std::vector<float> input_h = generateInput(n);

    std::vector<float> ref(n);
    scanSegmentsCPU(input_h.data(), ref.data(), n, segSize);

    std::vector<float> gpu;
    float ms = runKoggeStoneScan(input_h, gpu, segSize);

    bool ok = true;
    for (unsigned int i = 0; i < n; ++i) {
        if (!nearlyEqual(gpu[i], ref[i], 1e-2f)) {
            ok = false;
            printf("  mismatch at %u: cpu=%.6f gpu=%.6f\n", i, ref[i], gpu[i]);
            break;
        }
    }
    printf("segSize=%u blocks=%u N=%u: last=%.6f (cpu) / %.6f (gpu)  %.4f ms  [%s]\n",
           segSize, numBlocks, n, ref[n - 1], gpu[n - 1], ms, ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    bool ok = true;
    ok = runTestCase(64, 4) && ok;
    ok = runTestCase(256, 8) && ok;
    ok = runTestCase(1024, 4) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
