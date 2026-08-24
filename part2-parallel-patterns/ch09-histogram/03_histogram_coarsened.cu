// Chapter 9: Histogram
// §9.5  Thread coarsening (Fig. 9.14, interleaved partitioning)
//
// Privatization (file 02) reduces contention but adds overhead: each
// block's private histogram must be initialized and, at the end, merged
// into the public one -- "these initialize and commit operations are done
// once per thread block. Hence, the more thread blocks we use, the larger
// this overhead is" (§9.5). §9.5's fix is thread coarsening: launch fewer
// blocks and have each thread process multiple input pixels, so fewer
// private copies need to be initialized/committed.
//
// The book compares two coarsening strategies. Contiguous partitioning
// (Fig. 9.11/9.12) gives each thread a contiguous run of input pixels --
// simple, but "results in a sub-optimal memory access pattern" on a GPU
// because consecutive threads in a warp no longer touch consecutive
// addresses in the same iteration, defeating memory coalescing (§9.5).
// Interleaved partitioning (Fig. 9.13/9.14) instead has each thread jump by
// blockDim.x on every coarsening iteration, so "all threads jointly process
// the first blockDim.x elements of the block segment" in a fully coalesced
// access -- the strategy the book recommends for GPUs. This file implements
// the interleaved kernel (Fig. 9.14) directly, layered on top of file 02's
// shared-memory privatization exactly as the book does (Fig. 9.14's diff
// from Fig. 9.10 is only the coarsening loop, per §9.5's own description).

#include <cuda/atomic>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

#define NUM_BINS 256
#define COARSE_FACTOR 4

// ---------------------------------------------------------------------------
// §9.5, Fig. 9.14: privatized + coarsened histogram kernel (interleaved
// partitioning).
//
//   - bins_s init / __syncthreads(): identical to file 02.
//   - segment (Fig. 9.14 line 10): the offset of this block's input
//     segment, blockIdx.x * blockDim.x * COARSE_FACTOR pixels wide.
//   - Coarsening loop (lines 11-12): COARSE_FACTOR iterations; in
//     iteration c, thread threadIdx.x processes pixel
//     segment + c*blockDim.x + threadIdx.x. In iteration 0 all threads in
//     the block jointly cover the first blockDim.x pixels of the segment
//     (consecutive threadIdx.x -> consecutive addresses, so warp accesses
//     coalesce); in iteration 1 they jointly cover the next blockDim.x
//     pixels, and so on -- the "interleaved" access pattern of Fig. 9.13,
//     as opposed to contiguous partitioning where each thread would instead
//     own one *contiguous* run of COARSE_FACTOR pixels.
//   - Per-pixel body and commit phase: unchanged from file 02 (block-scope
//     atomic into bins_s during the loop, device-scope atomic to merge
//     nonzero bins_s entries into the public bins array afterward).
// ---------------------------------------------------------------------------
__global__ void histogram_coarsened_kernel(const unsigned char *image, unsigned int *bins, unsigned int width,
                                            unsigned int height) {
    __shared__ unsigned int bins_s[NUM_BINS];
    for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
        bins_s[b] = 0u;
    }
    __syncthreads();

    unsigned int count = width * height;
    unsigned int segment = blockIdx.x * blockDim.x * COARSE_FACTOR;
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
        unsigned int binVal = bins_s[b];
        if (binVal > 0) {
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bins_ref(bins[b]);
            bins_ref.fetch_add(binVal, cuda::memory_order_relaxed);
        }
    }
}

// CPU reference (Fig. 9.2): unchanged -- coarsening only changes the GPU
// work partitioning, not the sequential algorithm or its result.
void histogram_cpu(const unsigned char *image, unsigned int *bins, unsigned int width, unsigned int height) {
    for (unsigned int i = 0; i < width * height; ++i) {
        unsigned char b = image[i];
        ++bins[b];
    }
}

// Same synthetic-image generator as files 01/02.
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

// Runs the coarsened histogram kernel once (discarded warm-up launch
// first) and returns the timed kernel duration in ms. Grid size is
// divided by COARSE_FACTOR relative to files 01/02 since each thread now
// covers COARSE_FACTOR pixels.
float runCoarsenedHistogram(const unsigned char *image_h, unsigned int *bins_h, unsigned int width,
                             unsigned int height) {
    size_t count = static_cast<size_t>(width) * height;
    size_t imgBytes = count * sizeof(unsigned char);
    size_t binBytes = NUM_BINS * sizeof(unsigned int);

    unsigned char *image_d;
    unsigned int *bins_d;
    CUDA_CHECK(cudaMalloc((void **)&image_d, imgBytes));
    CUDA_CHECK(cudaMalloc((void **)&bins_d, binBytes));
    CUDA_CHECK(cudaMemcpy(image_d, image_h, imgBytes, cudaMemcpyHostToDevice));

    dim3 dimBlock(256);
    dim3 dimGrid((count + dimBlock.x * COARSE_FACTOR - 1) / (dimBlock.x * COARSE_FACTOR));

    CUDA_CHECK(cudaMemset(bins_d, 0, binBytes));
    histogram_coarsened_kernel<<<dimGrid, dimBlock>>>(image_d, bins_d, width, height);  // warm-up
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemset(bins_d, 0, binBytes));
    GpuTimer timer;
    timer.start();
    histogram_coarsened_kernel<<<dimGrid, dimBlock>>>(image_d, bins_d, width, height);
    CUDA_CHECK(cudaGetLastError());
    float ms = timer.stopAndGetMs();

    CUDA_CHECK(cudaMemcpy(bins_h, bins_d, binBytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(image_d));
    CUDA_CHECK(cudaFree(bins_d));

    return ms;
}

bool runTestCase(unsigned int width, unsigned int height) {
    size_t count = static_cast<size_t>(width) * height;
    std::vector<unsigned char> image_h = generateImage(count);
    std::vector<unsigned int> bins_ref(NUM_BINS, 0), bins_h(NUM_BINS, 0);

    histogram_cpu(image_h.data(), bins_ref.data(), width, height);
    float ms = runCoarsenedHistogram(image_h.data(), bins_h.data(), width, height);

    bool ok = true;
    for (int b = 0; b < NUM_BINS; ++b) {
        if (bins_h[b] != bins_ref[b]) {
            ok = false;
            fprintf(stderr, "Mismatch at bin %d: gpu=%u cpu=%u\n", b, bins_h[b], bins_ref[b]);
            break;
        }
    }

    printf("%ux%u (%zu pixels): %.3f ms  [%s]\n", width, height, count, ms, ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    bool ok = true;
    // Includes a case (1000x777) whose pixel count is not a multiple of
    // blockDim.x*COARSE_FACTOR, exercising the boundary check inside the
    // coarsening loop.
    ok = runTestCase(256, 256) && ok;
    ok = runTestCase(1000, 777) && ok;
    ok = runTestCase(1920, 1080) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
