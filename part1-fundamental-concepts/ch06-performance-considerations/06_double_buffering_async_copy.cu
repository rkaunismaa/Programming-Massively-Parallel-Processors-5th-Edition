// Chapter 6: Performance considerations
// §6.7  Double buffering, applied to shared-memory tile loading in the
// tiled matmul kernel, using hardware asynchronous copy (cp.async, exposed
// via the CUDA C++ standard library's cuda::memcpy_async / cuda::pipeline).
//
// The book's toy example (§6.7) removes a false write-after-read dependence
// by allocating two buffers, reading from inBuffer while writing to
// outBuffer, then swapping the two for the next iteration -- eliminating the
// second __syncthreads() that a single-buffer version would need between
// the read and the write.
//
// This file applies exactly that idea to the tile-loading step of the
// Chapter 5 tiled matmul (reimplemented locally, no cross-chapter include):
// two shared-memory buffers (Mds/Nds[2][...]) alternate ("ping-pong") across
// phases. But it goes one step further than the book's toy example, using
// SM80+ hardware asynchronous copy (cuda::memcpy_async, which lowers to the
// cp.async PTX instruction on Ampere and newer) so that the NEXT phase's
// tile is fetched directly from global memory into shared memory by the copy
// engine *while the current phase's TILE_WIDTH-deep inner product is still
// being computed on the tile already in the other buffer* -- true
// producer/consumer overlap between the async-copy engine and the compute
// pipeline, not just dependence elimination.
//
// cuda::memcpy_async targeting shared memory as its destination requires
// SM80 (Ampere) or newer; this file is compiled with -arch=sm_80 (per the
// chapter Makefile) and, at run time, checks the actual device's compute
// capability and exits gracefully with a clear message if it is below 8.0,
// rather than crashing with a cryptic "no kernel image available" error.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cooperative_groups.h>
#include <cuda/pipeline>

#include "../../common/cuda_utils.h"

namespace cg = cooperative_groups;

#define TILE_WIDTH 32

// ---------------------------------------------------------------------------
// §6.7: double-buffered tiled matmul with SM80 async-copy tile prefetch.
//
// Two physical tile buffers (index 0 and 1) ping-pong across phases. Each
// thread owns a thread-scoped cuda::pipeline that tracks its own two
// outstanding memcpy_async "batches" (the book's inBuffer/outBuffer split,
// realized here as a producer/consumer FIFO of depth 2):
//   - phase 0's tile load is issued before the loop starts (priming).
//   - at the top of iteration ph, phase (ph+1)'s tile load is issued into
//     the OTHER buffer *before* this thread waits for phase ph's own tile --
//     so the copy engine has the entire duration of phase ph's compute to
//     finish delivering phase ph+1's data.
//   - consumer_wait() blocks only until THIS thread's own oldest
//     outstanding copy lands; __syncthreads() is still required afterward so
//     that every thread in the block (whose tile elements were loaded by
//     other threads) can safely read the whole tile.
// ---------------------------------------------------------------------------
__global__ void matmulDoubleBufferedKernel(const float *__restrict__ M,
                                            const float *__restrict__ N,
                                            float *__restrict__ P, int Width) {
    __shared__ float Mds[2][TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[2][TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    int numPhases = Width / TILE_WIDTH;

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    auto issueLoad = [&](int ph, int buf) {
        pipe.producer_acquire();
        cuda::memcpy_async(&Mds[buf][ty][tx], &M[Row * Width + ph * TILE_WIDTH + tx],
                            sizeof(float), pipe);
        cuda::memcpy_async(&Nds[buf][ty][tx], &N[(ph * TILE_WIDTH + ty) * Width + Col],
                            sizeof(float), pipe);
        pipe.producer_commit();
    };

    // Prime the pipeline: issue phase 0's async load into buffer 0.
    issueLoad(0, 0);

    float Pvalue = 0.0f;
    for (int ph = 0; ph < numPhases; ++ph) {
        int buf = ph % 2;

        // Double buffering (§6.7): issue the NEXT phase's async load into
        // the OTHER buffer before consuming the current one, so the async
        // copy engine overlaps with this phase's compute below.
        if (ph + 1 < numPhases) {
            issueLoad(ph + 1, 1 - buf);
        }

        // Wait for this thread's own phase-ph copy to land, then a
        // block-wide barrier so every thread's tile elements (loaded by
        // other threads) are visible before any thread reads the tile.
        pipe.consumer_wait();
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[buf][ty][k] * Nds[buf][k][tx];
        }

        __syncthreads();
        pipe.consumer_release();
    }

    P[Row * Width + Col] = Pvalue;
}

void matrixMul_h(const float *M, const float *N, float *P, int Width) {
    for (int row = 0; row < Width; ++row) {
        for (int col = 0; col < Width; ++col) {
            float Pvalue = 0.0f;
            for (int k = 0; k < Width; ++k) {
                Pvalue += M[row * Width + k] * N[k * Width + col];
            }
            P[row * Width + col] = Pvalue;
        }
    }
}

int main() {
    // §6.7 note: cuda::memcpy_async targeting shared memory (cp.async) needs
    // SM80 (Ampere) or newer hardware. Check the device we actually landed
    // on and bail out cleanly rather than crash if it doesn't qualify -- this
    // keeps the sample portable to machines without an Ampere+ GPU.
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("Device %d: %s (compute capability %d.%d)\n", device, prop.name, prop.major, prop.minor);
    if (prop.major * 10 + prop.minor < 80) {
        printf("This sample requires compute capability >= 8.0 (Ampere or newer) for\n"
               "cuda::memcpy_async / cp.async hardware asynchronous copy support.\n"
               "Detected compute capability %d.%d on device %d (%s) -- skipping.\n",
               prop.major, prop.minor, device, prop.name);
        return 0;
    }

    const int Width = 1024;  // exact multiple of TILE_WIDTH
    if (Width % TILE_WIDTH != 0) {
        fprintf(stderr, "Width must be a multiple of TILE_WIDTH for this file\n");
        return 1;
    }

    size_t count = static_cast<size_t>(Width) * Width;
    size_t size = count * sizeof(float);

    std::vector<float> M_h(count), N_h(count), P_ref(count), P_h(count);

    for (size_t i = 0; i < count; ++i) {
        M_h[i] = static_cast<float>(i % 13) * 0.1f - 0.6f;
        N_h[i] = static_cast<float>(i % 7) * 0.2f - 0.6f;
    }

    printf("Computing CPU reference (Width=%d, %zu elements)...\n", Width, count);
    matrixMul_h(M_h.data(), N_h.data(), P_ref.data(), Width);

    float *M_d, *N_d, *P_d;
    CUDA_CHECK(cudaMalloc((void **)&M_d, size));
    CUDA_CHECK(cudaMalloc((void **)&N_d, size));
    CUDA_CHECK(cudaMalloc((void **)&P_d, size));
    CUDA_CHECK(cudaMemcpy(M_d, M_h.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(N_d, N_h.data(), size, cudaMemcpyHostToDevice));

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid(Width / TILE_WIDTH, Width / TILE_WIDTH, 1);

    // Warm up (discarded) before timing, so one-time PTX->SASS JIT cost
    // isn't attributed to the timed launch.
    matmulDoubleBufferedKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, Width);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    matmulDoubleBufferedKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, Width);
    CUDA_CHECK(cudaGetLastError());
    float ms = timer.stopAndGetMs();
    CUDA_CHECK(cudaMemcpy(P_h.data(), P_d, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(M_d));
    CUDA_CHECK(cudaFree(N_d));
    CUDA_CHECK(cudaFree(P_d));

    bool ok = true;
    for (size_t i = 0; i < count; ++i) {
        if (!nearlyEqual(P_h[i], P_ref[i])) {
            ok = false;
            fprintf(stderr, "Mismatch at i=%zu: gpu=%f cpu=%f\n", i, P_h[i], P_ref[i]);
            break;
        }
    }

    printf("Width = %d, TILE_WIDTH = %d\n", Width, TILE_WIDTH);
    printf("Double-buffered async-copy (§6.7) kernel time: %.3f ms  [%s]\n",
           ms, ok ? "match" : "MISMATCH");
    printf("%s\n", ok ? "PASS" : "FAIL");

    return ok ? 0 : 1;
}
