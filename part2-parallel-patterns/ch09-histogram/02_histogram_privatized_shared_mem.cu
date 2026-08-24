// Chapter 9: Histogram
// §9.4  Privatization (Fig. 9.10)
//
// §9.3 shows the basic kernel's atomic throughput is limited by memory
// latency: on a memory system with 200-cycle DRAM access latency, "the
// highest throughput one can achieve is one atomic operation every 400
// cycles ... 2.5 M atomics/second" for a *single* contended bin -- and
// real image histograms are heavily biased (Fig. 9.1's tree image skews
// toward bright bins), so a handful of bins absorb most of the contention.
//
// §9.4's fix is privatization: give each thread block its own private copy
// of the histogram so contention on any one bin is limited to threads in
// the same block (plus a modest amount of cross-block contention when
// private copies are merged into the public one at the end). The book
// presents this in two steps: Fig. 9.9 puts the private copies in global
// memory (one gridDim.x*NUM_BINS pool); Fig. 9.10 improves on that by
// moving each block's private copy into shared memory, since "any
// reduction in latency directly translates into improved throughput of
// atomic operations" and shared memory access latency is "a few cycles"
// vs. hundreds for DRAM (§9.4). This file implements Fig. 9.10 directly
// (the book's final, best privatized-only kernel): the private histogram
// is declared __shared__, giving every atomic increment during the main
// pass block scope (cuda::thread_scope_block) instead of device scope.

#include <cuda/atomic>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

#define NUM_BINS 256

// ---------------------------------------------------------------------------
// §9.4, Fig. 9.10: privatized histogram kernel, private copy in shared mem.
//
//   - bins_s (line 04): the block's private histogram, declared in shared
//     memory -- 256 unsigned ints = 1 KB, far under any block's shared
//     memory budget.
//   - Initialization (lines 05-07): each thread zeroes bins_s at a stride
//     of blockDim.x so the loop covers all NUM_BINS regardless of block
//     size, followed by __syncthreads() (line 08) so no thread starts
//     updating before every bin is zeroed.
//   - Main pass (lines 09-12): identical thread-per-pixel assignment as
//     file 01, but the atomic_ref now targets bins_s[b] with
//     cuda::thread_scope_block -- only threads in this block can contend
//     on it, and the narrower scope can be serviced at shared-memory
//     latency instead of device-wide DRAM/L2 latency.
//   - Commit phase (lines 13-21): __syncthreads() (line 13) lets every
//     thread finish updating bins_s before any commit begins; each thread
//     then walks a stride of the NUM_BINS private bins (line 15), skips
//     zero bins (line 16 -- avoids a wasted atomic when a bin got no hits
//     in this block), and atomically adds its nonzero private count into
//     the public bins array with cuda::thread_scope_device (lines 17-19),
//     since threads from *different* blocks can commit to the same public
//     bin simultaneously.
// ---------------------------------------------------------------------------
__global__ void histogram_privatized_kernel(const unsigned char *image, unsigned int *bins, unsigned int width,
                                             unsigned int height) {
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
        unsigned int binVal = bins_s[b];
        if (binVal > 0) {
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bins_ref(bins[b]);
            bins_ref.fetch_add(binVal, cuda::memory_order_relaxed);
        }
    }
}

// CPU reference (Fig. 9.2): unchanged from file 01, privatization is a
// GPU-side contention optimization and does not affect the sequential
// algorithm or its result.
void histogram_cpu(const unsigned char *image, unsigned int *bins, unsigned int width, unsigned int height) {
    for (unsigned int i = 0; i < width * height; ++i) {
        unsigned char b = image[i];
        ++bins[b];
    }
}

// Same synthetic-image generator as file 01: runs of identical pixels with
// Fig. 9.1's brightness distribution (6/12/14/32 per 64 -> black/dark
// gray/light gray/white), giving a realistic, contention-heavy input.
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

// Runs the privatized histogram kernel once (discarded warm-up launch
// first) and returns the timed kernel duration in ms.
float runPrivatizedHistogram(const unsigned char *image_h, unsigned int *bins_h, unsigned int width,
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
    dim3 dimGrid((count + dimBlock.x - 1) / dimBlock.x);

    CUDA_CHECK(cudaMemset(bins_d, 0, binBytes));
    histogram_privatized_kernel<<<dimGrid, dimBlock>>>(image_d, bins_d, width, height);  // warm-up
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemset(bins_d, 0, binBytes));
    GpuTimer timer;
    timer.start();
    histogram_privatized_kernel<<<dimGrid, dimBlock>>>(image_d, bins_d, width, height);
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
    float ms = runPrivatizedHistogram(image_h.data(), bins_h.data(), width, height);

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
    ok = runTestCase(256, 256) && ok;
    ok = runTestCase(1000, 777) && ok;
    ok = runTestCase(1920, 1080) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
