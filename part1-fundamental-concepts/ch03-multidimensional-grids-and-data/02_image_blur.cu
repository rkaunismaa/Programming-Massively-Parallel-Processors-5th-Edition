// Chapter 3: Multidimensional grids and data
// §3.3  Image blur -- a more complex kernel
//       -- blurKernel, Fig. 3.8; boundary handling, Fig. 3.9
//
// Each thread computes one output pixel as the average of the
// (2*BLUR_SIZE+1) x (2*BLUR_SIZE+1) patch of input pixels centered on
// (row, col) -- the same thread-to-output-pixel mapping as
// colorToGrayscaleConversion (§3.2), but each thread now does more work
// and must guard each individual patch access, not just its own pixel.
// We use BLUR_SIZE = 1, the book's worked 3x3 patch example (Fig. 3.7).

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

#define BLUR_SIZE 1  // radius: 2*BLUR_SIZE+1 = 3x3 patch (Fig. 3.7)

// ---------------------------------------------------------------------------
// §3.3, Fig. 3.8: blurKernel.
// The nested loop scans curRow = row-BLUR_SIZE .. row+BLUR_SIZE and
// curCol = col-BLUR_SIZE .. col+BLUR_SIZE over the patch centered on (row,
// col). The guard curRow>=0 && curRow<h && curCol>=0 && curCol<w (line 15 in
// the book's figure) excludes patch pixels that fall outside the image, e.g.
// near a corner or edge (Fig. 3.9); `pixels` counts how many patch pixels
// were actually valid so the average divides by the right count -- 9 for
// interior pixels, 6 on an edge, 4 at a corner, for a 3x3 patch.
// ---------------------------------------------------------------------------
__global__ void blurKernel(const unsigned char *in, unsigned char *out, int w, int h) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < w && row < h) {
        int pixVal = 0;
        int pixels = 0;

        for (int blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE; ++blurRow) {
            for (int blurCol = -BLUR_SIZE; blurCol <= BLUR_SIZE; ++blurCol) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;

                if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {
                    pixVal += in[curRow * w + curCol];
                    ++pixels;
                }
            }
        }

        out[row * w + col] = static_cast<unsigned char>(pixVal / pixels);
    }
}

// CPU reference: identical nested loop and boundary guard as the kernel.
void blur_h(const unsigned char *in, unsigned char *out, int w, int h) {
    for (int row = 0; row < h; ++row) {
        for (int col = 0; col < w; ++col) {
            int pixVal = 0;
            int pixels = 0;
            for (int blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE; ++blurRow) {
                for (int blurCol = -BLUR_SIZE; blurCol <= BLUR_SIZE; ++blurCol) {
                    int curRow = row + blurRow;
                    int curCol = col + blurCol;
                    if (curRow >= 0 && curRow < h && curCol >= 0 && curCol < w) {
                        pixVal += in[curRow * w + curCol];
                        ++pixels;
                    }
                }
            }
            out[row * w + col] = static_cast<unsigned char>(pixVal / pixels);
        }
    }
}

float runBlur(const unsigned char *in_h, unsigned char *out_h, int w, int h) {
    unsigned char *in_d, *out_d;
    int numPixels = w * h;

    CUDA_CHECK(cudaMalloc((void **)&in_d, numPixels * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc((void **)&out_d, numPixels * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemcpy(in_d, in_h, numPixels * sizeof(unsigned char), cudaMemcpyHostToDevice));

    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid((w + 15) / 16, (h + 15) / 16, 1);

    GpuTimer timer;
    timer.start();
    blurKernel<<<dimGrid, dimBlock>>>(in_d, out_d, w, h);
    CUDA_CHECK(cudaGetLastError());
    float ms = timer.stopAndGetMs();

    CUDA_CHECK(cudaMemcpy(out_h, out_d, numPixels * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(in_d));
    CUDA_CHECK(cudaFree(out_d));

    return ms;
}

int main() {
    // Deliberately not multiples of the 16x16 block size, so the run
    // exercises corner/edge pixels whose patches are clipped by the image
    // boundary (§3.3, Fig. 3.9), in addition to interior pixels.
    const int width = 403;
    const int height = 251;

    std::vector<unsigned char> in_h(static_cast<size_t>(width) * height);
    for (int i = 0; i < width * height; ++i) {
        in_h[i] = static_cast<unsigned char>((i * 37) % 256);
    }

    std::vector<unsigned char> out_ref(static_cast<size_t>(width) * height);
    std::vector<unsigned char> out_h(static_cast<size_t>(width) * height);

    // CPU reference (§3.3).
    blur_h(in_h.data(), out_ref.data(), width, height);

    // GPU version (§3.3, Fig. 3.8).
    float ms = runBlur(in_h.data(), out_h.data(), width, height);

    // All arithmetic here is integer (sum of unsigned char values divided by
    // an integer pixel count), so GPU and CPU results must match exactly.
    bool ok = true;
    for (int i = 0; i < width * height; ++i) {
        if (out_h[i] != out_ref[i]) {
            ok = false;
            fprintf(stderr, "Mismatch at pixel %d: gpu=%u cpu=%u\n", i,
                    static_cast<unsigned>(out_h[i]), static_cast<unsigned>(out_ref[i]));
            break;
        }
    }

    printf("image %dx%d (h x w), BLUR_SIZE=%d (%dx%d patch)\n",
           height, width, BLUR_SIZE, 2 * BLUR_SIZE + 1, 2 * BLUR_SIZE + 1);
    printf("GPU blurKernel time: %.3f ms\n", ms);
    printf("%s\n", ok ? "PASS" : "FAIL");

    return ok ? 0 : 1;
}
