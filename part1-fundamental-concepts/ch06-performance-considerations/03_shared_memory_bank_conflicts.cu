// Chapter 6: Performance considerations
// §6.4  Shared memory bank conflicts
//
// The book's worked example: with
//   __shared__ float a[TILE_DIM][TILE_DIM];   // TILE_DIM = 32
//   a[threadIdx.x][threadIdx.y] = ...;
// threads in the same warp (same threadIdx.y, consecutive threadIdx.x) write
// to linear indices threadIdx.x*32 + threadIdx.y, i.e. 0, 32, 64, ... --
// all landing in the same bank (32 mod 32 == 0), a 32-way bank conflict that
// the hardware must serialize.
//
// Padding the tile to
//   __shared__ float a[TILE_DIM][TILE_DIM + 1];
// changes the linear index to threadIdx.x*33 + threadIdx.y, so consecutive
// threadIdx.x values land in banks 0, 1, 2, ... (33 mod 32 == 1) -- a
// different bank per thread, conflict-free.
//
// This file embeds exactly that a[threadIdx.x][threadIdx.y] write pattern
// inside a full, verifiable shared-memory matrix-transpose kernel (the
// classic vehicle for this exact access pattern): each block stages a
// TILE_DIM x TILE_DIM input tile into shared memory with the strided
// a[threadIdx.x][threadIdx.y] store (conflicting, or padded/conflict-free),
// then writes it back out transposed with a coalesced global store.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

#define TILE_DIM 32

// ---------------------------------------------------------------------------
// §6.4: conflicting version. a[TILE_DIM][TILE_DIM] -- the shared-memory
// store a[threadIdx.x][threadIdx.y] = ... causes a 32-way bank conflict
// exactly as the book describes.
// ---------------------------------------------------------------------------
__global__ void transposeConflictKernel(const float *in, float *out, int Width) {
    __shared__ float a[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Coalesced global load (consecutive threadIdx.x -> consecutive x),
    // strided shared-memory store -- the book's conflicting pattern.
    a[threadIdx.x][threadIdx.y] = in[y * Width + x];
    __syncthreads();

    int xOut = blockIdx.y * TILE_DIM + threadIdx.x;
    int yOut = blockIdx.x * TILE_DIM + threadIdx.y;

    // Coalesced global store of the transposed tile.
    out[yOut * Width + xOut] = a[threadIdx.y][threadIdx.x];
}

// ---------------------------------------------------------------------------
// §6.4: padded, conflict-free version. a[TILE_DIM][TILE_DIM + 1] shifts each
// row start by one bank, eliminating the conflict.
// ---------------------------------------------------------------------------
__global__ void transposePaddedKernel(const float *in, float *out, int Width) {
    __shared__ float a[TILE_DIM][TILE_DIM + 1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    a[threadIdx.x][threadIdx.y] = in[y * Width + x];
    __syncthreads();

    int xOut = blockIdx.y * TILE_DIM + threadIdx.x;
    int yOut = blockIdx.x * TILE_DIM + threadIdx.y;

    out[yOut * Width + xOut] = a[threadIdx.y][threadIdx.x];
}

void transpose_h(const float *in, float *out, int Width) {
    for (int r = 0; r < Width; ++r) {
        for (int c = 0; c < Width; ++c) {
            out[c * Width + r] = in[r * Width + c];
        }
    }
}

int main() {
    const int Width = 4096;  // exact multiple of TILE_DIM (32)
    if (Width % TILE_DIM != 0) {
        fprintf(stderr, "Width must be a multiple of TILE_DIM for this file\n");
        return 1;
    }

    size_t count = static_cast<size_t>(Width) * Width;
    size_t size = count * sizeof(float);

    std::vector<float> in_h(count), out_ref(count);
    std::vector<float> out_conflict_h(count), out_padded_h(count);

    for (size_t i = 0; i < count; ++i) {
        in_h[i] = static_cast<float>(i % 997) * 0.01f - 5.0f;
    }

    printf("Computing CPU reference (Width=%d)...\n", Width);
    transpose_h(in_h.data(), out_ref.data(), Width);

    float *in_d, *out_d;
    CUDA_CHECK(cudaMalloc((void **)&in_d, size));
    CUDA_CHECK(cudaMalloc((void **)&out_d, size));
    CUDA_CHECK(cudaMemcpy(in_d, in_h.data(), size, cudaMemcpyHostToDevice));

    dim3 dimBlock(TILE_DIM, TILE_DIM, 1);
    dim3 dimGrid(Width / TILE_DIM, Width / TILE_DIM, 1);

    // Warm up both kernels once each (discarded) before timing.
    transposeConflictKernel<<<dimGrid, dimBlock>>>(in_d, out_d, Width);
    CUDA_CHECK(cudaGetLastError());
    transposePaddedKernel<<<dimGrid, dimBlock>>>(in_d, out_d, Width);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    transposeConflictKernel<<<dimGrid, dimBlock>>>(in_d, out_d, Width);
    CUDA_CHECK(cudaGetLastError());
    float conflict_ms = timer.stopAndGetMs();
    CUDA_CHECK(cudaMemcpy(out_conflict_h.data(), out_d, size, cudaMemcpyDeviceToHost));

    timer.start();
    transposePaddedKernel<<<dimGrid, dimBlock>>>(in_d, out_d, Width);
    CUDA_CHECK(cudaGetLastError());
    float padded_ms = timer.stopAndGetMs();
    CUDA_CHECK(cudaMemcpy(out_padded_h.data(), out_d, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(in_d));
    CUDA_CHECK(cudaFree(out_d));

    bool conflict_ok = true, padded_ok = true;
    for (size_t i = 0; i < count; ++i) {
        if (conflict_ok && !nearlyEqual(out_conflict_h[i], out_ref[i])) {
            conflict_ok = false;
            fprintf(stderr, "Conflict-kernel mismatch at i=%zu: gpu=%f cpu=%f\n",
                    i, out_conflict_h[i], out_ref[i]);
        }
        if (padded_ok && !nearlyEqual(out_padded_h[i], out_ref[i])) {
            padded_ok = false;
            fprintf(stderr, "Padded-kernel mismatch at i=%zu: gpu=%f cpu=%f\n",
                    i, out_padded_h[i], out_ref[i]);
        }
        if (!conflict_ok && !padded_ok) break;
    }

    printf("Width = %d, TILE_DIM = %d, dimBlock=(%d,%d,1), dimGrid=(%d,%d,1)\n",
           Width, TILE_DIM, dimBlock.x, dimBlock.y, dimGrid.x, dimGrid.y);
    printf("Conflicting a[%d][%d]     (§6.4, 32-way bank conflict) kernel time: %.3f ms  [%s]\n",
           TILE_DIM, TILE_DIM, conflict_ms, conflict_ok ? "match" : "MISMATCH");
    printf("Padded      a[%d][%d+1]   (§6.4, conflict-free)        kernel time: %.3f ms  [%s]\n",
           TILE_DIM, TILE_DIM, padded_ms, padded_ok ? "match" : "MISMATCH");
    printf("Speedup (conflict/padded): %.2fx\n", conflict_ms / padded_ms);

    bool ok = conflict_ok && padded_ok;
    printf("%s\n", ok ? "PASS" : "FAIL");

    return ok ? 0 : 1;
}
