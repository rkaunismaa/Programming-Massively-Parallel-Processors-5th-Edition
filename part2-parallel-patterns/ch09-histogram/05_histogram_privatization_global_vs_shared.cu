// Chapter 9: Histogram
// §9.4  Privatization -- global-memory private copies (Fig. 9.9) vs
// shared-memory private copies (Fig. 9.10)
//
// §9.4 presents privatization in two steps, as two complete kernel
// listings: Fig. 9.9 gives every thread block a private histogram
// allocated out of a global-memory pool (`bins_pool`, sized
// `gridDim.x * NUM_BINS`), reducing contention from device-wide to
// per-block; Fig. 9.10 then observes that "if the number of bins in the
// histogram is small enough, the private copy of the histogram can be
// declared in the block's shared memory... [which] has very short access
// latency (a few cycles). This reduced latency directly translates into
// [a] dramatic increase in the throughput of atomic operations" -- and
// gives Fig. 9.10 as the improved kernel. This repo's file 02 already
// implements Fig. 9.10; Fig. 9.9 itself was never given its own sample,
// so the book's own before/after argument was never actually measured.
// This file implements Fig. 9.9 and times it directly against the same
// Fig. 9.10 kernel (duplicated here per this chapter's established
// self-contained-file convention) on identical input.
//
// The two kernels are otherwise structurally identical -- same
// one-thread-per-pixel update pass, same per-block commit-nonzero-bins
// loop -- differing only in where the private histogram lives and how
// its lifetime is managed:
//   - Fig. 9.9: bins_priv points into a global-memory pool the host
//     allocates with gridDim.x * NUM_BINS elements and must zero before
//     each launch (the figure only shows the kernel; zeroing a
//     global-memory buffer between calls is an ordinary host
//     responsibility, not shown in the listing). Private-histogram
//     updates use cuda::thread_scope_block, matching Fig. 9.9's own text
//     even though the memory itself is physically global -- the *scope*
//     only needs to cover the threads that can see the same bins_priv
//     pointer (this block's threads), not the whole device.
//   - Fig. 9.10: bins_priv is a __shared__ array, zeroed cooperatively by
//     the block itself (no host-side buffer or initialization needed).

#include <cuda/atomic>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

#define NUM_BINS 256

// ---------------------------------------------------------------------------
// §9.4, Fig. 9.9: histogram kernel with private bins per block in GLOBAL
// memory. bins_pool must be zeroed by the host before this kernel runs.
// ---------------------------------------------------------------------------
__global__ void histogram_privatized_global_kernel(const unsigned char *image, unsigned int *bins,
                                                     unsigned int *bins_pool, unsigned int width,
                                                     unsigned int height) {
    unsigned int *bins_priv = &bins_pool[blockIdx.x * NUM_BINS];

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width * height) {
        unsigned char b = image[i];
        cuda::atomic_ref<unsigned int, cuda::thread_scope_block> bins_priv_ref(bins_priv[b]);
        bins_priv_ref.fetch_add(1, cuda::memory_order_relaxed);
    }
    __syncthreads();

    for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
        if (bins_priv[b] > 0) {
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bins_ref(bins[b]);
            bins_ref.fetch_add(bins_priv[b], cuda::memory_order_relaxed);
        }
    }
}

// ---------------------------------------------------------------------------
// §9.4, Fig. 9.10: histogram kernel with private bins per block in SHARED
// memory (identical to 02_histogram_privatized_shared_mem.cu, duplicated
// here for a direct, same-binary timing comparison).
// ---------------------------------------------------------------------------
__global__ void histogram_privatized_shared_kernel(const unsigned char *image, unsigned int *bins,
                                                     unsigned int width, unsigned int height) {
    __shared__ unsigned int bins_s[NUM_BINS];
    for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
        bins_s[b] = 0u;
    }
    __syncthreads();

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width * height) {
        unsigned char b = image[i];
        cuda::atomic_ref<unsigned int, cuda::thread_scope_block> bins_s_ref(bins_s[b]);
        bins_s_ref.fetch_add(1, cuda::memory_order_relaxed);
    }
    __syncthreads();

    for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
        if (bins_s[b] > 0) {
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bins_ref(bins[b]);
            bins_ref.fetch_add(bins_s[b], cuda::memory_order_relaxed);
        }
    }
}

void histogram_cpu(const unsigned char *image, unsigned int *bins, unsigned int width, unsigned int height) {
    for (unsigned int i = 0; i < width * height; ++i) {
        unsigned char b = image[i];
        ++bins[b];
    }
}

// Same synthetic-image generator as files 01-04.
std::vector<unsigned char> generateImage(size_t count) {
    std::vector<unsigned char> image(count);
    size_t i = 0;
    unsigned int state = 12345u;
    while (i < count) {
        state = state * 1103515245u + 12345u;
        unsigned int runLen = 1u + ((state >> 16) % 24u);
        state = state * 1103515245u + 12345u;
        unsigned int r = (state >> 8) % 100u;
        unsigned int val;
        if (r < 9)
            val = state % 64u;
        else if (r < 28)
            val = 64u + (state % 64u);
        else if (r < 50)
            val = 128u + (state % 64u);
        else
            val = 192u + (state % 64u);
        for (unsigned int k = 0; k < runLen && i < count; ++k, ++i) {
            image[i] = static_cast<unsigned char>(val);
        }
    }
    return image;
}

bool runTestCase(unsigned int width, unsigned int height) {
    size_t count = static_cast<size_t>(width) * height;
    size_t imgBytes = count * sizeof(unsigned char);
    size_t binBytes = NUM_BINS * sizeof(unsigned int);

    std::vector<unsigned char> image_h = generateImage(count);
    std::vector<unsigned int> bins_ref(NUM_BINS, 0);
    histogram_cpu(image_h.data(), bins_ref.data(), width, height);

    unsigned char *image_d;
    unsigned int *binsGlobal_d, *binsShared_d, *binsPool_d;
    CUDA_CHECK(cudaMalloc((void **)&image_d, imgBytes));
    CUDA_CHECK(cudaMalloc((void **)&binsGlobal_d, binBytes));
    CUDA_CHECK(cudaMalloc((void **)&binsShared_d, binBytes));
    CUDA_CHECK(cudaMemcpy(image_d, image_h.data(), imgBytes, cudaMemcpyHostToDevice));

    dim3 dimBlock(256);
    dim3 dimGrid((count + dimBlock.x - 1) / dimBlock.x);
    size_t poolBytes = static_cast<size_t>(dimGrid.x) * NUM_BINS * sizeof(unsigned int);
    CUDA_CHECK(cudaMalloc((void **)&binsPool_d, poolBytes));

    // Warm-up launches (discarded), so PTX->SASS JIT cost isn't folded into
    // either timed measurement below.
    CUDA_CHECK(cudaMemset(binsPool_d, 0, poolBytes));
    CUDA_CHECK(cudaMemset(binsGlobal_d, 0, binBytes));
    histogram_privatized_global_kernel<<<dimGrid, dimBlock>>>(image_d, binsGlobal_d, binsPool_d, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemset(binsShared_d, 0, binBytes));
    histogram_privatized_shared_kernel<<<dimGrid, dimBlock>>>(image_d, binsShared_d, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;

    CUDA_CHECK(cudaMemset(binsPool_d, 0, poolBytes));
    CUDA_CHECK(cudaMemset(binsGlobal_d, 0, binBytes));
    timer.start();
    histogram_privatized_global_kernel<<<dimGrid, dimBlock>>>(image_d, binsGlobal_d, binsPool_d, width, height);
    CUDA_CHECK(cudaGetLastError());
    float globalMs = timer.stopAndGetMs();

    CUDA_CHECK(cudaMemset(binsShared_d, 0, binBytes));
    timer.start();
    histogram_privatized_shared_kernel<<<dimGrid, dimBlock>>>(image_d, binsShared_d, width, height);
    CUDA_CHECK(cudaGetLastError());
    float sharedMs = timer.stopAndGetMs();

    std::vector<unsigned int> binsGlobal_h(NUM_BINS), binsShared_h(NUM_BINS);
    CUDA_CHECK(cudaMemcpy(binsGlobal_h.data(), binsGlobal_d, binBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(binsShared_h.data(), binsShared_d, binBytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(image_d));
    CUDA_CHECK(cudaFree(binsGlobal_d));
    CUDA_CHECK(cudaFree(binsShared_d));
    CUDA_CHECK(cudaFree(binsPool_d));

    bool globalOk = true, sharedOk = true;
    for (int b = 0; b < NUM_BINS; ++b) {
        if (globalOk && binsGlobal_h[b] != bins_ref[b]) {
            globalOk = false;
            fprintf(stderr, "Global-privatized mismatch at bin %d: gpu=%u cpu=%u\n", b,
                    binsGlobal_h[b], bins_ref[b]);
        }
        if (sharedOk && binsShared_h[b] != bins_ref[b]) {
            sharedOk = false;
            fprintf(stderr, "Shared-privatized mismatch at bin %d: gpu=%u cpu=%u\n", b,
                    binsShared_h[b], bins_ref[b]);
        }
    }

    printf("%ux%u (%zu pixels): global-priv (Fig. 9.9) %.3f ms  [%s]  |  "
           "shared-priv (Fig. 9.10) %.3f ms  [%s]\n",
           width, height, count, globalMs, globalOk ? "match" : "MISMATCH", sharedMs,
           sharedOk ? "match" : "MISMATCH");

    return globalOk && sharedOk;
}

int main() {
    bool ok = true;
    ok = runTestCase(256, 256) && ok;
    ok = runTestCase(1000, 777) && ok;
    ok = runTestCase(1920, 1080) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
