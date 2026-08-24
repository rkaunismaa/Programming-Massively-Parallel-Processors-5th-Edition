// Chapter 11: Scan
// §11.9  Consolidating block segments for a global scan
//        (Fig. 11.14 - 11.18)
//
// Every kernel so far (files 01-05) only scans WITHIN a block's own
// segment; blocks never talk to each other. Real inputs can be millions or
// billions of elements, needing a grid-wide (global) scan across many
// blocks, which requires an INTER-BLOCK scan: each block's segment total
// must be combined with the running total of all preceding blocks.
//
// The book surveys three ways to do this and this file implements the one
// it ultimately builds real kernel code for:
//
//   1. Three-kernel scan-scan-add (naive): local scan / scan of block sums
//      / add pass, as three separate kernel launches. §11.9 derives its
//      exact traffic: (4 + 8/S)*N bytes for segment size S, ~16 B/element
//      for large S -- half the peak 419e9 elements/s the book calculates in
//      §11.8 for an ideal (H100) scan, because the N intermediate scanned
//      values are written out and read back an extra time.
//   2. Three-kernel reduce-scan-scan: replaces the first kernel's local
//      SCAN with a REDUCTION (fewer writes: N/S values instead of N), for
//      (3 + 3/S)*N bytes, ~12 B/element -- better, but still loads the
//      input from global memory twice overall.
//   3. Single-kernel, in-kernel inter-block scan via SINGLE LOOKBACK
//      (Fig. 11.16(b), Fig. 11.17, Fig. 11.18) -- the one implemented here.
//      Avoids a second global-memory pass entirely by having each block
//      "look back" to exactly the immediately preceding block's finished
//      partial sum, using UNIDIRECTIONAL synchronization (a block only
//      waits for an EARLIER block, never a later one) instead of a
//      grid-wide barrier.
//
// Two things make single-lookback work correctly within one kernel launch:
//
//   - DYNAMIC block-index assignment (Fig. 11.18, lines 04-10): a block's
//     logical index `bid` is NOT blockIdx.x. Instead, the first thread of
//     each block atomically increments a global counter the instant the
//     block starts executing, and that count becomes `bid`. This guarantees
//     that a block with a LOWER bid started executing (and therefore is
//     making forward progress) no later than any block with a higher bid --
//     which is exactly what rules out deadlock: no block ever waits on a
//     block that hasn't even started yet.
//   - Device-scope atomics with acquire/release ordering (Fig. 11.17): each
//     block's leader thread waits on the PRECEDING block's flag
//     (`cuda::memory_order_acquire`, so the subsequent read of that block's
//     partial sum can't be reordered before the flag check), then publishes
//     its own partial sum and sets its own flag
//     (`cuda::memory_order_release`, so the flag update can't be reordered
//     before the partial-sum write becomes visible). `cuda::thread_scope_
//     device` is required because the flag is read and written by threads
//     in DIFFERENT blocks, not just different warps of the same block.
//
// The book notes single lookback's own weakness -- an N/S-long chain of
// unidirectional syncs becomes a long critical path for many blocks -- and
// describes (but leaves as an exercise) DECOUPLED lookback, where a block
// looks further back until it finds an already-fully-scanned sum. That
// generalization is out of scope here; this file implements exactly what
// Fig. 11.17/11.18 present.
//
// The block-local part of the kernel (load / per-thread sequential scan in
// registers / block-level scan of thread sums / add) is, per the book,
// "mostly the same as the code in Fig. 11.13" -- i.e. it reuses file 05's
// register-tiled coarsened scan, with the inter-block add folded in before
// the final store.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda/atomic>

#include "../../common/cuda_utils.h"

#define WARP_SIZE 32
#define COARSE_FACTOR 4
#define BLOCK_DIM 256

__device__ unsigned int warpIdx() { return threadIdx.x / WARP_SIZE; }
__device__ unsigned int laneIdx() { return threadIdx.x % WARP_SIZE; }

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
// §11.9, Fig. 11.17: inter-block scan via single lookback. Only the block's
// leader thread (last thread in the block) participates in the lookback
// chain; all threads receive the result via shared memory.
// ---------------------------------------------------------------------------
__device__ float interBlockScan(float val, unsigned int bid, float *partialSums, unsigned int *flags) {
    __shared__ float previousSum;

    if (threadIdx.x == blockDim.x - 1) {
        if (bid == 0) {
            previousSum = 0.0f;
        } else {
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> flagRef(flags[bid - 1]);
            while (flagRef.fetch_add(0u, cuda::memory_order_acquire) == 0u) {
                // spin until the preceding block publishes its partial sum
            }
            previousSum = partialSums[bid - 1];
        }
        partialSums[bid] = previousSum + val;

        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> myFlagRef(flags[bid]);
        myFlagRef.fetch_add(1u, cuda::memory_order_release);
    }
    __syncthreads();

    return previousSum;
}

// ---------------------------------------------------------------------------
// §11.9, Fig. 11.18: single-kernel global scan with dynamic block-index
// assignment and single-lookback inter-block scan.
// ---------------------------------------------------------------------------
__global__ void globalScanKernel(const float *input, float *output, unsigned int N,
                                  float *partialSums, unsigned int *flags, unsigned int *blockCounter) {
    __shared__ unsigned int bid_s;
    if (threadIdx.x == 0) {
        bid_s = atomicAdd(blockCounter, 1u);
    }
    __syncthreads();
    unsigned int bid = bid_s;

    __shared__ float buffer_s[COARSE_FACTOR * BLOCK_DIM];
    __shared__ float warpSums_s[BLOCK_DIM / WARP_SIZE];
    __shared__ float scannedThreadSums_s[BLOCK_DIM];

    unsigned int segment = bid * blockDim.x * COARSE_FACTOR;

    // Coalesced load.
    for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        unsigned int idx = segment + c * BLOCK_DIM + threadIdx.x;
        buffer_s[c * BLOCK_DIM + threadIdx.x] = (idx < N) ? input[idx] : 0.0f;
    }
    __syncthreads();

    // Register-tiled per-thread sequential scan (Fig. 11.13, file 05).
    float buffer_r[COARSE_FACTOR];
    unsigned int threadSegment = threadIdx.x * COARSE_FACTOR;
    buffer_r[0] = buffer_s[threadSegment];
#pragma unroll
    for (unsigned int c = 1; c < COARSE_FACTOR; ++c) {
        buffer_r[c] = buffer_r[c - 1] + buffer_s[threadSegment + c];
    }

    // Block-level scan of the per-thread subsegment sums.
    float threadSum = buffer_r[COARSE_FACTOR - 1];
    float scannedThreadSum = blockScan(threadSum, warpSums_s);
    scannedThreadSums_s[threadIdx.x] = scannedThreadSum;
    __syncthreads();

    float prevThread = (threadIdx.x > 0) ? scannedThreadSums_s[threadIdx.x - 1] : 0.0f;
#pragma unroll
    for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        buffer_s[threadSegment + c] = buffer_r[c] + prevThread;
    }
    __syncthreads();
    // Local (per-block) scan complete: buffer_s now holds the block's fully
    // scanned segment, matching what file 05's kernel would produce alone.

    // Inter-block scan: the block's total is the last element of its
    // finished local scan.
    float blockSum = buffer_s[COARSE_FACTOR * BLOCK_DIM - 1];
    float previousBlockSum = interBlockScan(blockSum, bid, partialSums, flags);

    // Add pass: fold in every preceding block's total (Fig. 11.18, lines
    // 45-48; a no-op for block 0, where previousBlockSum is exactly 0).
#pragma unroll
    for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        buffer_s[threadSegment + c] += previousBlockSum;
    }
    __syncthreads();

    // Coalesced store.
    for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        unsigned int idx = segment + c * BLOCK_DIM + threadIdx.x;
        if (idx < N) {
            output[idx] = buffer_s[c * BLOCK_DIM + threadIdx.x];
        }
    }
}

// True GLOBAL sequential reference: single accumulator over the whole
// array, since this file consolidates every block's segment.
void scanCPU(const float *input, float *output, unsigned int N) {
    float acc = 0.0f;
    for (unsigned int i = 0; i < N; ++i) {
        acc += input[i];
        output[i] = acc;
    }
}

std::vector<float> generateInput(unsigned int n) {
    std::vector<float> v(n);
    unsigned int state = 623456789u;
    for (unsigned int i = 0; i < n; ++i) {
        state = state * 1103515245u + 12345u;
        v[i] = static_cast<float>((state >> 8) & 0xFFFFFF) / static_cast<float>(0x1000000);
    }
    return v;
}

float runGlobalScan(const std::vector<float> &input_h, std::vector<float> &output_h) {
    unsigned int n = static_cast<unsigned int>(input_h.size());
    size_t bytes = n * sizeof(float);
    unsigned int segSize = COARSE_FACTOR * BLOCK_DIM;
    unsigned int numBlocks = (n + segSize - 1) / segSize;

    float *input_d, *output_d, *partialSums_d;
    unsigned int *flags_d, *blockCounter_d;
    CUDA_CHECK(cudaMalloc((void **)&input_d, bytes));
    CUDA_CHECK(cudaMalloc((void **)&output_d, bytes));
    CUDA_CHECK(cudaMalloc((void **)&partialSums_d, numBlocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&flags_d, numBlocks * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc((void **)&blockCounter_d, sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(input_d, input_h.data(), bytes, cudaMemcpyHostToDevice));

    dim3 dimBlock(BLOCK_DIM);
    dim3 dimGrid(numBlocks);

    auto resetState = [&]() {
        CUDA_CHECK(cudaMemset(flags_d, 0, numBlocks * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemset(blockCounter_d, 0, sizeof(unsigned int)));
    };

    resetState();
    globalScanKernel<<<dimGrid, dimBlock>>>(input_d, output_d, n, partialSums_d, flags_d, blockCounter_d);  // warm-up
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    resetState();
    GpuTimer timer;
    timer.start();
    globalScanKernel<<<dimGrid, dimBlock>>>(input_d, output_d, n, partialSums_d, flags_d, blockCounter_d);
    CUDA_CHECK(cudaGetLastError());
    float ms = timer.stopAndGetMs();

    output_h.resize(n);
    CUDA_CHECK(cudaMemcpy(output_h.data(), output_d, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(input_d));
    CUDA_CHECK(cudaFree(output_d));
    CUDA_CHECK(cudaFree(partialSums_d));
    CUDA_CHECK(cudaFree(flags_d));
    CUDA_CHECK(cudaFree(blockCounter_d));
    return ms;
}

bool runTestCase(unsigned int n) {
    std::vector<float> input_h = generateInput(n);

    std::vector<float> ref(n);
    scanCPU(input_h.data(), ref.data(), n);

    std::vector<float> gpu;
    float ms = runGlobalScan(input_h, gpu);

    unsigned int segSize = COARSE_FACTOR * BLOCK_DIM;
    unsigned int numBlocks = (n + segSize - 1) / segSize;

    bool ok = true;
    for (unsigned int i = 0; i < n; ++i) {
        if (!nearlyEqual(gpu[i], ref[i], 1e-2f)) {
            ok = false;
            printf("  mismatch at %u: cpu=%.6f gpu=%.6f\n", i, ref[i], gpu[i]);
            break;
        }
    }
    printf("N=%u (blocks=%u): last=%.6f (cpu) / %.6f (gpu)  %.4f ms  [%s]\n",
           n, numBlocks, ref[n - 1], gpu[n - 1], ms, ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    bool ok = true;
    ok = runTestCase(4096) && ok;              // 4 blocks
    ok = runTestCase(100000) && ok;             // 98 blocks, last block partial
    ok = runTestCase(1 << 20) && ok;            // 1024 blocks

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
