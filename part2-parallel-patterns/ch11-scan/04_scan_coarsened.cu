// Chapter 11: Scan
// §11.6  Coarsening to improve work-efficiency (Fig. 11.11, Fig. 11.12)
//
// §11.5 showed that Kogge-Stone-style parallel scan is fundamentally NOT
// work efficient: O(N*log2(N)) additions vs. the sequential algorithm's
// O(N). The fix, as with reduction in Chapter 10, is thread coarsening: do
// more of the work in a work-efficient SEQUENTIAL scan per thread, and only
// pay the less-efficient parallel-scan overhead once, on a much smaller
// array (one value per thread instead of one value per input element).
//
// This again uses the scan-scan-add decomposition (Fig. 11.6), now at the
// THREAD granularity within a block: partition the block's segment
// (COARSE_FACTOR*BLOCK_DIM elements) into BLOCK_DIM contiguous
// subsegments of COARSE_FACTOR elements each, one per thread. Each thread
// sequentially scans its own subsegment (work efficient, no synchronization
// needed). Each thread's final subsegment value (the subsegment's sum) then
// participates in a block-level scan (file 03's warp-primitive blockScan,
// which Fig. 11.12 explicitly reuses from Fig. 11.9). Finally, each thread
// adds the scanned sum of all PRECEDING threads to every element of its own
// subsegment.
//
// A subtlety Fig. 11.12 calls out explicitly: loading COARSE_FACTOR
// elements per thread must NOT be done by each thread reading its own
// contiguous subsegment directly from global memory -- consecutive threads
// would then touch global addresses COARSE_FACTOR apart, which is
// uncoalesced. Instead, the block's segment is loaded in COARSE_FACTOR
// coalesced chunks of BLOCK_DIM elements (chunk c, thread t loads global
// element `segment + c*BLOCK_DIM + t`), landing every element at its own
// unique position within the block's segment in shared memory regardless of
// load order; the SAME shared-memory array is then re-addressed by each
// thread as `threadIdx.x*COARSE_FACTOR + c` to access its own contiguous
// subsegment for the sequential scan. The same trick, symmetrically, is
// used for the final coalesced store back to global memory.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#include "../../common/cuda_utils.h"

#define WARP_SIZE 32
#define COARSE_FACTOR 4
#define BLOCK_DIM 256

__device__ unsigned int warpIdx() { return threadIdx.x / WARP_SIZE; }
__device__ unsigned int laneIdx() { return threadIdx.x % WARP_SIZE; }

// Same warp-level / block-level scan device functions as file 03 (Figs.
// 11.8, 11.9) -- duplicated here per this project's self-contained-file
// convention.
__device__ float warpScan(float val) {
    unsigned int lane = laneIdx();
    for (unsigned int stride = 1; stride < WARP_SIZE; stride *= 2) {
        float temp = __shfl_up_sync(0xffffffff, val, stride);
        if (lane >= stride) {
            val += temp;
        }
    }
    return val;
}

__device__ float blockScan(float val, float *warpSums_s) {
    unsigned int lane = laneIdx();
    unsigned int warp = warpIdx();
    unsigned int numWarps = blockDim.x / WARP_SIZE;

    val = warpScan(val);
    if (lane == WARP_SIZE - 1) {
        warpSums_s[warp] = val;
    }
    __syncthreads();

    if (warp == 0) {
        float warpSumVal = (lane < numWarps) ? warpSums_s[lane] : 0.0f;
        warpSumVal = warpScan(warpSumVal);
        if (lane < numWarps) {
            warpSums_s[lane] = warpSumVal;
        }
    }
    __syncthreads();

    if (warp > 0) {
        val += warpSums_s[warp - 1];
    }
    return val;
}

// ---------------------------------------------------------------------------
// §11.6, Fig. 11.12: thread-coarsened scan-scan-add kernel. Each block
// covers COARSE_FACTOR*BLOCK_DIM elements; each thread owns COARSE_FACTOR of
// them.
// ---------------------------------------------------------------------------
__global__ void coarsenedScanKernel(const float *input, float *output, unsigned int N) {
    __shared__ float buffer_s[COARSE_FACTOR * BLOCK_DIM];
    __shared__ float warpSums_s[BLOCK_DIM / WARP_SIZE];
    __shared__ float scannedThreadSums_s[BLOCK_DIM];

    unsigned int segment = blockIdx.x * blockDim.x * COARSE_FACTOR;

    // Coalesced load: COARSE_FACTOR chunks of BLOCK_DIM contiguous elements.
    for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        unsigned int idx = segment + c * BLOCK_DIM + threadIdx.x;
        buffer_s[c * BLOCK_DIM + threadIdx.x] = (idx < N) ? input[idx] : 0.0f;
    }
    __syncthreads();

    // Each thread sequentially (work-efficiently) scans its own contiguous
    // subsegment, re-addressing the same shared buffer thread-contiguously.
    unsigned int threadSegment = threadIdx.x * COARSE_FACTOR;
    for (unsigned int c = 1; c < COARSE_FACTOR; ++c) {
        buffer_s[threadSegment + c] += buffer_s[threadSegment + c - 1];
    }

    // Block-level scan of the per-thread subsegment sums.
    float threadSum = buffer_s[threadSegment + COARSE_FACTOR - 1];
    float scannedThreadSum = blockScan(threadSum, warpSums_s);
    scannedThreadSums_s[threadIdx.x] = scannedThreadSum;
    __syncthreads();

    // Add the sum of all preceding threads' elements to this thread's
    // subsegment.
    if (threadIdx.x > 0) {
        float prev = scannedThreadSums_s[threadIdx.x - 1];
        for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
            buffer_s[threadSegment + c] += prev;
        }
    }
    __syncthreads();

    // Coalesced store, symmetric with the coalesced load above.
    for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        unsigned int idx = segment + c * BLOCK_DIM + threadIdx.x;
        if (idx < N) {
            output[idx] = buffer_s[c * BLOCK_DIM + threadIdx.x];
        }
    }
}

// Sequential reference, per segment (see file 01 for rationale). Segment
// size here is COARSE_FACTOR*BLOCK_DIM.
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
    unsigned int state = 423456789u;
    for (unsigned int i = 0; i < n; ++i) {
        state = state * 1103515245u + 12345u;
        v[i] = static_cast<float>((state >> 8) & 0xFFFFFF) / static_cast<float>(0x1000000);
    }
    return v;
}

float runCoarsenedScan(const std::vector<float> &input_h, std::vector<float> &output_h,
                        unsigned int numBlocks) {
    unsigned int n = static_cast<unsigned int>(input_h.size());
    size_t bytes = n * sizeof(float);

    float *input_d, *output_d;
    CUDA_CHECK(cudaMalloc((void **)&input_d, bytes));
    CUDA_CHECK(cudaMalloc((void **)&output_d, bytes));
    CUDA_CHECK(cudaMemcpy(input_d, input_h.data(), bytes, cudaMemcpyHostToDevice));

    dim3 dimBlock(BLOCK_DIM);
    dim3 dimGrid(numBlocks);

    coarsenedScanKernel<<<dimGrid, dimBlock>>>(input_d, output_d, n);  // warm-up
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    coarsenedScanKernel<<<dimGrid, dimBlock>>>(input_d, output_d, n);
    CUDA_CHECK(cudaGetLastError());
    float ms = timer.stopAndGetMs();

    output_h.resize(n);
    CUDA_CHECK(cudaMemcpy(output_h.data(), output_d, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(input_d));
    CUDA_CHECK(cudaFree(output_d));
    return ms;
}

bool runTestCase(unsigned int numBlocks) {
    unsigned int segSize = COARSE_FACTOR * BLOCK_DIM;
    unsigned int n = segSize * numBlocks;
    std::vector<float> input_h = generateInput(n);

    std::vector<float> ref(n);
    scanSegmentsCPU(input_h.data(), ref.data(), n, segSize);

    std::vector<float> gpu;
    float ms = runCoarsenedScan(input_h, gpu, numBlocks);

    bool ok = true;
    for (unsigned int i = 0; i < n; ++i) {
        if (!nearlyEqual(gpu[i], ref[i], 1e-2f)) {
            ok = false;
            printf("  mismatch at %u: cpu=%.6f gpu=%.6f\n", i, ref[i], gpu[i]);
            break;
        }
    }
    printf("blocks=%u N=%u (COARSE_FACTOR=%d, BLOCK_DIM=%d): last=%.6f (cpu) / %.6f (gpu)  %.4f ms  [%s]\n",
           numBlocks, n, COARSE_FACTOR, BLOCK_DIM, ref[n - 1], gpu[n - 1], ms, ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    bool ok = true;
    ok = runTestCase(1) && ok;
    ok = runTestCase(4) && ok;
    ok = runTestCase(16) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
