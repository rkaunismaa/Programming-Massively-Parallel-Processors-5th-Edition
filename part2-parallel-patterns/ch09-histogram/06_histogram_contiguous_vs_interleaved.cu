// Chapter 9: Histogram
// §9.5  Thread coarsening -- contiguous partitioning (Fig. 9.12) vs
// interleaved partitioning (Fig. 9.14)
//
// §9.5 presents two complete kernel listings for assigning multiple input
// pixels to a coarsened thread and states a clear preference: "contiguous
// partitioning on GPUs results in a sub-optimal memory access pattern...
// we need to make sure that threads in a warp access consecutive
// locations to enable memory coalescing. This observation motivates
// interleaved partitioning." This repo's file 03 already implements the
// interleaved kernel (Fig. 9.14); the contiguous kernel (Fig. 9.12) was
// never given its own sample, so the book's own coalescing argument was
// never actually measured. This file implements both (duplicating file
// 03's interleaved kernel here per this chapter's self-contained-file
// convention) and times them directly against each other on identical
// input.
//
// The two kernels are identical in every respect -- same shared-memory
// privatized histogram, same COARSE_FACTOR, same commit phase -- except
// for a single line, the per-iteration pixel index:
//   - Fig. 9.12 (contiguous): i = segment + threadIdx.x*COARSE_FACTOR + c.
//     For a fixed c, consecutive threadIdx.x values are COARSE_FACTOR
//     apart in memory -- each thread instead walks a small *contiguous*
//     run of COARSE_FACTOR pixels that is entirely its own, non-consecutive
//     with its neighbors' runs.
//   - Fig. 9.14 (interleaved): i = segment + c*blockDim.x + threadIdx.x.
//     For a fixed c, consecutive threadIdx.x values are 1 apart in memory
//     -- coalesced.

#include <cuda/atomic>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

#define NUM_BINS 256
#define COARSE_FACTOR 4

// ---------------------------------------------------------------------------
// §9.5, Fig. 9.12: histogram kernel with coarsening using CONTIGUOUS
// partitioning. Each thread processes COARSE_FACTOR pixels that are
// consecutive in memory -- non-coalesced across a warp.
// ---------------------------------------------------------------------------
__global__ void histogram_contiguous_kernel(const unsigned char *image, unsigned int *bins,
                                             unsigned int width, unsigned int height) {
    __shared__ unsigned int bins_s[NUM_BINS];
    for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
        bins_s[b] = 0u;
    }
    __syncthreads();

    unsigned int count = width * height;
    unsigned int segment = COARSE_FACTOR * blockIdx.x * blockDim.x;
    for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        unsigned int i = segment + threadIdx.x * COARSE_FACTOR + c;
        if (i < count) {
            unsigned char b = image[i];
            cuda::atomic_ref<unsigned int, cuda::thread_scope_block> bins_s_ref(bins_s[b]);
            bins_s_ref.fetch_add(1, cuda::memory_order_relaxed);
        }
    }
    __syncthreads();

    for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
        if (bins_s[b] > 0) {
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bins_ref(bins[b]);
            bins_ref.fetch_add(bins_s[b], cuda::memory_order_relaxed);
        }
    }
}

// ---------------------------------------------------------------------------
// §9.5, Fig. 9.14: histogram kernel with coarsening using INTERLEAVED
// partitioning (identical to 03_histogram_coarsened.cu, duplicated here
// for a direct, same-binary timing comparison). Coalesced across a warp.
// ---------------------------------------------------------------------------
__global__ void histogram_interleaved_kernel(const unsigned char *image, unsigned int *bins,
                                              unsigned int width, unsigned int height) {
    __shared__ unsigned int bins_s[NUM_BINS];
    for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
        bins_s[b] = 0u;
    }
    __syncthreads();

    unsigned int count = width * height;
    unsigned int segment = COARSE_FACTOR * blockIdx.x * blockDim.x;
    for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        unsigned int i = segment + c * blockDim.x + threadIdx.x;
        if (i < count) {
            unsigned char b = image[i];
            cuda::atomic_ref<unsigned int, cuda::thread_scope_block> bins_s_ref(bins_s[b]);
            bins_s_ref.fetch_add(1, cuda::memory_order_relaxed);
        }
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

// Same synthetic-image generator as files 01-05.
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
    unsigned int *binsContig_d, *binsInterleaved_d;
    CUDA_CHECK(cudaMalloc((void **)&image_d, imgBytes));
    CUDA_CHECK(cudaMalloc((void **)&binsContig_d, binBytes));
    CUDA_CHECK(cudaMalloc((void **)&binsInterleaved_d, binBytes));
    CUDA_CHECK(cudaMemcpy(image_d, image_h.data(), imgBytes, cudaMemcpyHostToDevice));

    dim3 dimBlock(256);
    dim3 dimGrid((count + dimBlock.x * COARSE_FACTOR - 1) / (dimBlock.x * COARSE_FACTOR));

    // Warm-up launches (discarded).
    CUDA_CHECK(cudaMemset(binsContig_d, 0, binBytes));
    histogram_contiguous_kernel<<<dimGrid, dimBlock>>>(image_d, binsContig_d, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemset(binsInterleaved_d, 0, binBytes));
    histogram_interleaved_kernel<<<dimGrid, dimBlock>>>(image_d, binsInterleaved_d, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;

    CUDA_CHECK(cudaMemset(binsContig_d, 0, binBytes));
    timer.start();
    histogram_contiguous_kernel<<<dimGrid, dimBlock>>>(image_d, binsContig_d, width, height);
    CUDA_CHECK(cudaGetLastError());
    float contigMs = timer.stopAndGetMs();

    CUDA_CHECK(cudaMemset(binsInterleaved_d, 0, binBytes));
    timer.start();
    histogram_interleaved_kernel<<<dimGrid, dimBlock>>>(image_d, binsInterleaved_d, width, height);
    CUDA_CHECK(cudaGetLastError());
    float interleavedMs = timer.stopAndGetMs();

    std::vector<unsigned int> binsContig_h(NUM_BINS), binsInterleaved_h(NUM_BINS);
    CUDA_CHECK(cudaMemcpy(binsContig_h.data(), binsContig_d, binBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(
        cudaMemcpy(binsInterleaved_h.data(), binsInterleaved_d, binBytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(image_d));
    CUDA_CHECK(cudaFree(binsContig_d));
    CUDA_CHECK(cudaFree(binsInterleaved_d));

    bool contigOk = true, interleavedOk = true;
    for (int b = 0; b < NUM_BINS; ++b) {
        if (contigOk && binsContig_h[b] != bins_ref[b]) {
            contigOk = false;
            fprintf(stderr, "Contiguous mismatch at bin %d: gpu=%u cpu=%u\n", b, binsContig_h[b],
                    bins_ref[b]);
        }
        if (interleavedOk && binsInterleaved_h[b] != bins_ref[b]) {
            interleavedOk = false;
            fprintf(stderr, "Interleaved mismatch at bin %d: gpu=%u cpu=%u\n", b,
                    binsInterleaved_h[b], bins_ref[b]);
        }
    }

    printf("%ux%u (%zu pixels): contiguous (Fig. 9.12) %.3f ms  [%s]  |  "
           "interleaved (Fig. 9.14) %.3f ms  [%s]\n",
           width, height, count, contigMs, contigOk ? "match" : "MISMATCH", interleavedMs,
           interleavedOk ? "match" : "MISMATCH");

    return contigOk && interleavedOk;
}

int main() {
    bool ok = true;
    ok = runTestCase(256, 256) && ok;
    ok = runTestCase(1000, 777) && ok;
    ok = runTestCase(1920, 1080) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
