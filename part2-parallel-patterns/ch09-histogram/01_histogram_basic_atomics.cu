// Chapter 9: Histogram
// §9.2  Atomic operations and a basic histogram kernel (Fig. 9.6)
//
// A histogram is a display of occurrences of data values in a data set
// (§9.1). This chapter's running example, per the book's own figure (Fig.
// 9.1, "the histogram of pixel intensity values of the grayscale image of a
// tree"), is a grayscale image's pixel-intensity histogram, not a text
// histogram -- §9.1: "Grayscale images typically consist of pixels with
// intensity values ranging between 0 and 255 where 0 is black and 255 is
// white." Fig. 9.2's sequential C reference treats "each histogram bin
// represents a single pixel value" (§9.2 commentary on Fig. 9.2), so this
// file uses 256 bins, one per possible unsigned char pixel value.
//
// Unlike every parallel pattern in Ch. 2-8, histogram output does NOT
// follow the owner-computes rule: many threads can update the very same
// bin, an "output interference" that a race condition (Figs. 9.4-9.5) can
// corrupt. §9.2's fix is the CUDA C++ atomic-reference API: construct a
// cuda::atomic_ref<T, scope> over the target location and call its
// fetch_add method, which performs the read-modify-write as an indivisible
// unit. This file is the *basic* kernel (Fig. 9.6): one thread per input
// pixel, every atomic add goes straight to the public/global bins array
// with cuda::thread_scope_device (serialized across the whole GPU) -- the
// baseline that 02/03/04 progressively optimize via privatization,
// coarsening, and thread-level privatization.

#include <cuda/atomic>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

#define NUM_BINS 256

// ---------------------------------------------------------------------------
// §9.2, Fig. 9.6: basic histogram kernel.
//
//   - i (Fig. 9.6 line 03): global thread index, one thread per input pixel,
//     replacing the sequential for-loop of the CPU version (Fig. 9.2).
//   - Boundary check (line 04): threads past the end of the image do nothing.
//   - bins_ref (line 06): a cuda::atomic_ref<unsigned int,
//     cuda::thread_scope_device> constructed over bins[b] -- device scope
//     because increments to the same bin must be serialized whether they
//     come from threads in the same block or different blocks.
//   - fetch_add(1, cuda::memory_order_relaxed) (line 07): atomically adds 1
//     to bins[b]. Relaxed ordering suffices because the only other memory
//     access this thread performs -- the read of image[i] that produced b --
//     already has an instruction-level data dependency on the atomic's
//     target address, so the hardware cannot reorder them (§9.2).
// ---------------------------------------------------------------------------
__global__ void histogram_kernel(const unsigned char *image, unsigned int *bins, unsigned int width,
                                  unsigned int height) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width * height) {
        unsigned char b = image[i];
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bins_ref(bins[b]);
        bins_ref.fetch_add(1, cuda::memory_order_relaxed);
    }
}

// CPU reference, Fig. 9.2: sequential pass over the image, ++bins[b] per
// pixel. O(N) and memory-bound on a CPU per §9.1's commentary.
void histogram_cpu(const unsigned char *image, unsigned int *bins, unsigned int width, unsigned int height) {
    for (unsigned int i = 0; i < width * height; ++i) {
        unsigned char b = image[i];
        ++bins[b];
    }
}

// Synthetic grayscale image generator. Produces runs of identical pixel
// values (simulating the "large patches of pixels of identical value" that
// §9.6 calls out for pictures of the sky) whose brightness distribution
// mirrors Fig. 9.1's own worked example: of every 64 pixels, 6 are black
// [0-63] (9.4%), 12 are dark gray [64-127] (18.8%), 14 are light gray
// [128-191] (21.9%), and 32 are white [192-255] (50.0%) -- "pixels that are
// heavily concentrated in the bright (higher) intensity intervals" (§9.1).
// This also gives files 01-04 a shared, realistic, contention-heavy input:
// heavy bias toward a few bins is exactly what §9.3 says drives up
// contention in the naive kernel and motivates privatization in §9.4.
std::vector<unsigned char> generateImage(size_t count) {
    std::vector<unsigned char> image(count);
    size_t i = 0;
    unsigned int state = 12345u;
    while (i < count) {
        state = state * 1103515245u + 12345u;
        unsigned int runLen = 1u + ((state >> 16) % 24u);  // 1-24 identical pixels per run
        state = state * 1103515245u + 12345u;
        unsigned int r = (state >> 8) % 100u;
        unsigned int val;
        if (r < 9)
            val = state % 64u;               // black (trunk): [0-63],   ~9.4%
        else if (r < 28)
            val = 64u + (state % 64u);        // dark gray (leaves): [64-127], ~18.8%
        else if (r < 50)
            val = 128u + (state % 64u);       // light gray (grass): [128-191], ~21.9%
        else
            val = 192u + (state % 64u);       // white (sky): [192-255], ~50.0%
        for (unsigned int k = 0; k < runLen && i < count; ++k, ++i) {
            image[i] = static_cast<unsigned char>(val);
        }
    }
    return image;
}

// Runs the basic histogram kernel once (with a discarded warm-up launch
// first, so PTX->SASS JIT cost isn't folded into the timed measurement) and
// returns the timed kernel duration in ms.
float runBasicHistogram(const unsigned char *image_h, unsigned int *bins_h, unsigned int width, unsigned int height) {
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
    histogram_kernel<<<dimGrid, dimBlock>>>(image_d, bins_d, width, height);  // warm-up
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemset(bins_d, 0, binBytes));
    GpuTimer timer;
    timer.start();
    histogram_kernel<<<dimGrid, dimBlock>>>(image_d, bins_d, width, height);
    CUDA_CHECK(cudaGetLastError());
    float ms = timer.stopAndGetMs();

    CUDA_CHECK(cudaMemcpy(bins_h, bins_d, binBytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(image_d));
    CUDA_CHECK(cudaFree(bins_d));

    return ms;
}

// Runs one width x height test case: builds a deterministic synthetic
// image, computes the CPU reference histogram, launches the kernel, and
// checks the 256 bin counts match exactly (integer counts, so exact
// equality rather than nearlyEqual).
bool runTestCase(unsigned int width, unsigned int height) {
    size_t count = static_cast<size_t>(width) * height;
    std::vector<unsigned char> image_h = generateImage(count);
    std::vector<unsigned int> bins_ref(NUM_BINS, 0), bins_h(NUM_BINS, 0);

    histogram_cpu(image_h.data(), bins_ref.data(), width, height);
    float ms = runBasicHistogram(image_h.data(), bins_h.data(), width, height);

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
    // Mix of image sizes that are and aren't multiples of the 256-thread
    // block, so partial blocks at the end of the image are exercised too.
    ok = runTestCase(256, 256) && ok;    // exact multiple of block size
    ok = runTestCase(1000, 777) && ok;   // not a multiple
    ok = runTestCase(1920, 1080) && ok;  // realistic image size

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
