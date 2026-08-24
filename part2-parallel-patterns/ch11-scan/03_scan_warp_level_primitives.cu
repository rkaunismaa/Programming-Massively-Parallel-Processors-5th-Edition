// Chapter 11: Scan
// §11.4  Warp-level primitives to reduce synchronization
//        (Fig. 11.6, 11.7, 11.8, 11.9, 11.10)
//
// File 02's double-buffered kernel still pays one __syncthreads() and three
// shared-memory accesses per loop iteration, for a single floating-point
// addition -- synchronization and shared-memory latency dominate. §11.4's
// fix: decompose the block-level scan into per-WARP scans that use
// shuffle-based warp-level primitives (register-to-register, no shared
// memory, no __syncthreads()), then combine the warps' results.
//
// The combination method is the SCAN-SCAN-ADD decomposition (Fig. 11.6):
//   1. scan each of several smaller segments independently
//   2. scan the array of per-segment sums (the last element of each
//      segment's local scan result)
//   3. add each segment's entry in the scanned-sums array (excluding its
//      own contribution, i.e. the PRECEDING segment's scanned sum) to every
//      element of that segment
// Fig. 11.7 applies this at the warp granularity: every warp does a
// warp-level Kogge-Stone scan (Fig. 11.8) on its own contiguous 32-element
// slice of the block segment; the last lane of each warp holds that warp's
// sum, which gets written to a small shared array of per-warp sums; warp 0
// then scans THAT array (again via the same warp-level primitive); finally
// every warp except warp 0 adds its entry in the scanned warp-sums array to
// all of its own elements (Fig. 11.9's blockScan device function, used by
// the kernel in Fig. 11.10).
//
// Only two __syncthreads() calls total per block now (after warps write
// their sums, and after warp 0 finishes scanning them) instead of one per
// loop iteration.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#include "../../common/cuda_utils.h"

#define WARP_SIZE 32

__device__ unsigned int warpIdx() { return threadIdx.x / WARP_SIZE; }
__device__ unsigned int laneIdx() { return threadIdx.x % WARP_SIZE; }

// ---------------------------------------------------------------------------
// §11.4, Fig. 11.8: warp-level inclusive scan via __shfl_up_sync. For lanes
// where the shuffle source would be out of range (lane < stride), CUDA
// clamps the shuffle to return the calling thread's own value, so `temp` is
// always well-defined; we simply don't add it in for those lanes.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// §11.4, Fig. 11.9: block-level scan via the scan-scan-add decomposition
// over warp-level scans. `warpSums_s` must have room for blockDim.x/32
// floats (the caller allocates it).
// ---------------------------------------------------------------------------
__device__ float blockScan(float val, float *warpSums_s) {
    unsigned int lane = laneIdx();
    unsigned int warp = warpIdx();
    unsigned int numWarps = blockDim.x / WARP_SIZE;

    val = warpScan(val);  // stage 1: independent per-warp scans

    if (lane == WARP_SIZE - 1) {
        warpSums_s[warp] = val;  // last lane of each warp holds the warp's sum
    }
    __syncthreads();

    if (warp == 0) {
        float warpSumVal = (lane < numWarps) ? warpSums_s[lane] : 0.0f;
        warpSumVal = warpScan(warpSumVal);  // stage 2: scan of warp sums
        if (lane < numWarps) {
            warpSums_s[lane] = warpSumVal;
        }
    }
    __syncthreads();

    if (warp > 0) {
        val += warpSums_s[warp - 1];  // stage 3: add preceding warps' total
    }
    return val;
}

// ---------------------------------------------------------------------------
// §11.4, Fig. 11.10: the block-level scan kernel is now just load / blockScan
// / store.
// ---------------------------------------------------------------------------
__global__ void warpPrimitiveScanKernel(const float *input, float *output, unsigned int N) {
    extern __shared__ float warpSums_s[];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (i < N) ? input[i] : 0.0f;
    val = blockScan(val, warpSums_s);
    if (i < N) {
        output[i] = val;
    }
}

// Sequential reference, per segment (see file 01 for rationale).
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
    unsigned int state = 323456789u;
    for (unsigned int i = 0; i < n; ++i) {
        state = state * 1103515245u + 12345u;
        v[i] = static_cast<float>((state >> 8) & 0xFFFFFF) / static_cast<float>(0x1000000);
    }
    return v;
}

float runWarpPrimitiveScan(const std::vector<float> &input_h, std::vector<float> &output_h,
                            unsigned int segSize) {
    unsigned int n = static_cast<unsigned int>(input_h.size());
    size_t bytes = n * sizeof(float);
    unsigned int numWarps = segSize / WARP_SIZE;
    size_t shmemBytes = numWarps * sizeof(float);

    float *input_d, *output_d;
    CUDA_CHECK(cudaMalloc((void **)&input_d, bytes));
    CUDA_CHECK(cudaMalloc((void **)&output_d, bytes));
    CUDA_CHECK(cudaMemcpy(input_d, input_h.data(), bytes, cudaMemcpyHostToDevice));

    dim3 dimBlock(segSize);
    dim3 dimGrid(n / segSize);

    warpPrimitiveScanKernel<<<dimGrid, dimBlock, shmemBytes>>>(input_d, output_d, n);  // warm-up
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    warpPrimitiveScanKernel<<<dimGrid, dimBlock, shmemBytes>>>(input_d, output_d, n);
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
    float ms = runWarpPrimitiveScan(input_h, gpu, segSize);

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
    // segSize must be a multiple of WARP_SIZE (32) for the warp decomposition.
    ok = runTestCase(64, 4) && ok;
    ok = runTestCase(256, 8) && ok;
    ok = runTestCase(1024, 4) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
