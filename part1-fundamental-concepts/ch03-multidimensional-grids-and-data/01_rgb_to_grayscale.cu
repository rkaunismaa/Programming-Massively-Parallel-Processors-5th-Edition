// Chapter 3: Multidimensional grids and data
// §3.1  Multidimensional grid organization -- conceptual background: grids and
//       blocks are 3D (dim3), and blockIdx/threadIdx/gridDim/blockDim give a
//       thread its coordinates. See the chapter README for a summary; §3.1
//       has no standalone code listing of its own.
// §3.2  Mapping threads to multidimensional data
//       -- colorToGrayscaleConversion kernel, Fig. 3.4
//
// A picture is a natural fit for a 2D grid of 2D blocks (§3.2, Fig. 3.2): one
// thread is mapped to one output pixel via
//   row = blockIdx.y*blockDim.y + threadIdx.y   (vertical coordinate)
//   col = blockIdx.x*blockDim.x + threadIdx.x   (horizontal coordinate)
// The input image Pin is RGB (CHANNELS=3 bytes/pixel), linearized in
// row-major order; the output Pout is single-channel grayscale.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

#define CHANNELS 3

// ---------------------------------------------------------------------------
// §3.2, Fig. 3.4: colorToGrayscaleConversion kernel.
// grayOffset = row*width+col is the linearized index of the output pixel
// (one byte each). rgbOffset = grayOffset*CHANNELS is the start of the three
// consecutive input bytes (r, g, b) for that pixel. The conversion formula
// is the book's L = 0.299*r + 0.587*g + 0.114*b (§3.2).
// ---------------------------------------------------------------------------
__global__ void colorToGrayscaleConversion(const unsigned char *Pin, unsigned char *Pout,
                                            int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // §3.2: only threads whose col and row are both within range participate
    // (extra threads exist because gridDim*blockDim is rounded up to a
    // multiple of the 16x16 block size; see Fig. 3.5's boundary discussion).
    if (col < width && row < height) {
        int grayOffset = row * width + col;
        int rgbOffset = grayOffset * CHANNELS;
        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];
        Pout[grayOffset] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

// CPU reference: identical formula and loop order to the kernel.
void colorToGrayscaleConversion_h(const unsigned char *Pin, unsigned char *Pout,
                                   int width, int height) {
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            int grayOffset = row * width + col;
            int rgbOffset = grayOffset * CHANNELS;
            unsigned char r = Pin[rgbOffset];
            unsigned char g = Pin[rgbOffset + 1];
            unsigned char b = Pin[rgbOffset + 2];
            Pout[grayOffset] = static_cast<unsigned char>(0.299f * r + 0.587f * g + 0.114f * b);
        }
    }
}

float runGrayscale(const unsigned char *Pin_h, unsigned char *Pout_h, int width, int height) {
    int numPixels = width * height;
    unsigned char *Pin_d, *Pout_d;

    CUDA_CHECK(cudaMalloc((void **)&Pin_d, numPixels * CHANNELS * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc((void **)&Pout_d, numPixels * sizeof(unsigned char)));

    CUDA_CHECK(cudaMemcpy(Pin_d, Pin_h, numPixels * CHANNELS * sizeof(unsigned char),
                           cudaMemcpyHostToDevice));

    // §3.2: dimBlock is fixed at 16x16; dimGrid is sized by ceiling division
    // of (width, height) by (16, 16) so the grid covers every pixel, e.g.
    //   dim3 dimGrid(ceil(m/16.0), ceil(n/16.0), 1);
    //   dim3 dimBlock(16, 16, 1);
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid((width + 15) / 16, (height + 15) / 16, 1);

    GpuTimer timer;
    timer.start();
    colorToGrayscaleConversion<<<dimGrid, dimBlock>>>(Pin_d, Pout_d, width, height);
    CUDA_CHECK(cudaGetLastError());
    float ms = timer.stopAndGetMs();

    CUDA_CHECK(cudaMemcpy(Pout_h, Pout_d, numPixels * sizeof(unsigned char),
                           cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(Pin_d));
    CUDA_CHECK(cudaFree(Pout_d));

    return ms;
}

int main() {
    // Deliberately not multiples of the 16x16 block size, so the run
    // exercises the boundary blocks described in §3.2 / Fig. 3.5.
    const int width = 403;
    const int height = 251;

    std::vector<unsigned char> Pin_h(static_cast<size_t>(width) * height * CHANNELS);
    for (int i = 0; i < width * height; ++i) {
        Pin_h[i * CHANNELS + 0] = static_cast<unsigned char>((i * 7) % 256);
        Pin_h[i * CHANNELS + 1] = static_cast<unsigned char>((i * 13) % 256);
        Pin_h[i * CHANNELS + 2] = static_cast<unsigned char>((i * 29) % 256);
    }

    std::vector<unsigned char> Pout_ref(static_cast<size_t>(width) * height);
    std::vector<unsigned char> Pout_h(static_cast<size_t>(width) * height);

    // CPU reference (§3.2 formula).
    colorToGrayscaleConversion_h(Pin_h.data(), Pout_ref.data(), width, height);

    // GPU version (§3.2, Fig. 3.4).
    float ms = runGrayscale(Pin_h.data(), Pout_h.data(), width, height);

    // Single-precision multiply-adds may round slightly differently between
    // host and device (e.g. due to FMA contraction), so allow the converted
    // byte to differ by at most 1 from the CPU reference.
    bool ok = true;
    for (int i = 0; i < width * height; ++i) {
        int diff = static_cast<int>(Pout_h[i]) - static_cast<int>(Pout_ref[i]);
        if (diff < -1 || diff > 1) {
            ok = false;
            fprintf(stderr, "Mismatch at pixel %d: gpu=%u cpu=%u\n", i,
                    static_cast<unsigned>(Pout_h[i]), static_cast<unsigned>(Pout_ref[i]));
            break;
        }
    }

    dim3 dimGrid((width + 15) / 16, (height + 15) / 16, 1);
    printf("image %dx%d (h x w), dimBlock=(16,16,1), dimGrid=(%d,%d,1)\n",
           height, width, dimGrid.x, dimGrid.y);
    printf("GPU colorToGrayscaleConversion time: %.3f ms\n", ms);
    printf("%s\n", ok ? "PASS" : "FAIL");

    return ok ? 0 : 1;
}
