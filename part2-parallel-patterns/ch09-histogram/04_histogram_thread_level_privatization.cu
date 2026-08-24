// Chapter 9: Histogram
// §9.6  Thread-level privatization (Fig. 9.15)
//
// File 03 privatizes at the block level (a shared-memory histogram) and at
// the thread level only in the trivial sense that each thread does its own
// bin lookups. §9.6 adds a third level of privatization: "since each thread
// processes multiple input elements after coarsening, we can create a copy
// of the histogram that is private to the thread which the thread can
// update without using atomic operations" -- but only for the *single*
// most-recently-seen bin, since privatizing all 256 bins per thread would
// be "prohibitively expensive" (§9.6). This pays off specifically for data
// with local runs of identical values -- exactly the kind of image data
// this chapter's example uses ("in pictures of the sky, there can be large
// patches of pixels of identical value", §9.6) and exactly what this
// chapter's synthetic test image contains by construction (runs of
// identical pixels, shared with files 01-03).
//
// Fig. 9.15's mechanism (§9.6 prose): a thread loads its first pixel and
// sets a thread-private run counter to 1 for that pixel's bin. On each
// subsequent coarsening iteration it loads the next pixel; if it falls in
// the *same* bin as the one currently being tracked, the thread merely
// increments its private counter (no atomic operation at all). If it falls
// in a *different* bin, the thread commits its private counter's
// accumulated value to the block's shared-memory histogram with a single
// atomic add, then starts tracking the new bin with counter reset to 1.
// "With this scheme, the update is always at least one element behind" --
// so after the coarsening loop exits, one final atomic commits whatever
// bin/count the thread was still tracking (§9.6).

#include <cuda/atomic>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

#define NUM_BINS 256
#define COARSE_FACTOR 4

// ---------------------------------------------------------------------------
// §9.6, Fig. 9.15: privatized + coarsened + thread-level-privatized
// histogram kernel.
//
//   - bins_s init / __syncthreads(): identical to files 02/03.
//   - First pixel (lines 11-13): boundary-checked load of this thread's
//     first pixel (interleaved index, c=0) into currVal, with the
//     thread-private run counter accum initialized to 1.
//   - Coarsening loop from c=1 (line 14): mirrors file 03's interleaved
//     index arithmetic, but starts at c=1 since c=0 was already loaded
//     above.
//   - Same-bin fast path (lines 17-19): if the newly loaded pixel bNext
//     equals currVal, no atomic operation happens at all -- just
//     ++accum, a plain register increment.
//   - Different-bin commit (lines 21-25): otherwise the thread atomically
//     adds its accumulated run (accum) to bins_s[currVal] once (instead of
//     one atomic per pixel in the run), then starts tracking bNext with
//     accum reset to 1.
//   - Final commit (lines 29-31): after the loop, the thread's last
//     tracked run (currVal/accum) has not yet been committed -- "the
//     update is always at least one element behind" -- so one more atomic
//     add closes it out. Guarded by the same boundary check as the first
//     pixel load, so threads that never had a valid first pixel skip this.
//   - Commit phase (unchanged from files 02/03): merge nonzero bins_s
//     entries into the public bins array with a device-scope atomic.
// ---------------------------------------------------------------------------
__global__ void histogram_thread_privatized_kernel(const unsigned char *image, unsigned int *bins,
                                                     unsigned int width, unsigned int height) {
    __shared__ unsigned int bins_s[NUM_BINS];
    for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
        bins_s[b] = 0u;
    }
    __syncthreads();

    unsigned int count = width * height;
    unsigned int segment = blockIdx.x * blockDim.x * COARSE_FACTOR;
    unsigned int i = segment + threadIdx.x;  // c = 0

    if (i < count) {
        unsigned char currVal = image[i];
        unsigned int accum = 1;

        for (unsigned int c = 1; c < COARSE_FACTOR; ++c) {
            i = segment + c * blockDim.x + threadIdx.x;
            if (i < count) {
                unsigned char bNext = image[i];
                if (bNext == currVal) {
                    ++accum;  // same run: plain register increment, no atomic
                } else {
                    // run ended: commit the accumulated count for currVal in
                    // one atomic add, then start tracking the new bin
                    cuda::atomic_ref<unsigned int, cuda::thread_scope_block> bins_s_ref(bins_s[currVal]);
                    bins_s_ref.fetch_add(accum, cuda::memory_order_relaxed);
                    currVal = bNext;
                    accum = 1;
                }
            }
        }

        // The last tracked run has not been committed yet -- commit it now.
        cuda::atomic_ref<unsigned int, cuda::thread_scope_block> bins_s_ref(bins_s[currVal]);
        bins_s_ref.fetch_add(accum, cuda::memory_order_relaxed);
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

// CPU reference (Fig. 9.2): unchanged -- thread-level privatization only
// changes how the GPU batches its atomic updates, not the result.
void histogram_cpu(const unsigned char *image, unsigned int *bins, unsigned int width, unsigned int height) {
    for (unsigned int i = 0; i < width * height; ++i) {
        unsigned char b = image[i];
        ++bins[b];
    }
}

// Same synthetic-image generator as files 01-03: runs of identical pixels
// (mean run length ~12.5) with Fig. 9.1's brightness distribution. The
// runs are what let this kernel's same-bin fast path actually trigger.
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

// Runs the thread-level-privatized histogram kernel once (discarded
// warm-up launch first) and returns the timed kernel duration in ms.
float runThreadPrivatizedHistogram(const unsigned char *image_h, unsigned int *bins_h, unsigned int width,
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
    histogram_thread_privatized_kernel<<<dimGrid, dimBlock>>>(image_d, bins_d, width, height);  // warm-up
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemset(bins_d, 0, binBytes));
    GpuTimer timer;
    timer.start();
    histogram_thread_privatized_kernel<<<dimGrid, dimBlock>>>(image_d, bins_d, width, height);
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
    float ms = runThreadPrivatizedHistogram(image_h.data(), bins_h.data(), width, height);

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
    // Includes a case (1000x777) not a multiple of blockDim.x*COARSE_FACTOR,
    // exercising boundary checks on both the first-pixel load and the
    // coarsening loop.
    ok = runTestCase(256, 256) && ok;
    ok = runTestCase(1000, 777) && ok;
    ok = runTestCase(1920, 1080) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
