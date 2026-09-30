// Chapter 5: Memory architecture and data locality
// §5.6  Impact of memory usage on occupancy
//
// §5.6 works a specific numeric example: "consider if a kernel has thread
// blocks that use 38 KB of shared memory, and each have 256 threads. In
// this case, the kernel uses an average of (38 KB)/(256 threads) = 152
// B/thread of shared memory. With such shared memory usage, the kernel
// cannot achieve full occupancy. Each SM can only host a maximum of
// (228 KB)/(152 B/thread) = 1536 threads. Therefore, the maximum achievable
// occupancy of this kernel will be (1536 assigned threads)/(2048 maximum
// threads) = 75%" (using the H100's 228 KB shared memory/SM and 2048
// threads/SM as its example limits). This is the same kind of resource
// tradeoff Chapter 4 covered for *registers* -- and this repo's Chapter 4
// occupancy sample (04_occupancy_query.cu) already exercises
// cudaOccupancyMaxActiveBlocksPerMultiprocessor() for register pressure --
// this sample exercises the identical API, but for *shared memory*
// pressure, reproducing this section's specific lesson.
//
// Rather than hardcode the book's H100 figures (this repo has no H100 to
// verify them against), this sample queries the actual GPU's limits via
// cudaGetDeviceProperties() and asks the real occupancy API for the real
// answer. To isolate the shared-memory effect the way the book's own
// example does, block size is held fixed at 256 threads (the book's own
// value) while the *hypothetical* dynamic shared memory requested per
// block is swept from 0 up to the architecture's classic 48 KB default
// per-block limit -- cudaOccupancyMaxActiveBlocksPerMultiprocessor()
// accepts any dynamicSMemSize query without requiring a kernel actually
// launched with that value, so this cleanly varies only the shared-memory
// dimension, not thread count.
//
// The kernel queried is 03_dynamic_shared_tiled_matmul.cu's
// matrixMulTiledDynamicSharedKernel at tileWidth=16 (16*16 == 256 threads,
// matching the book's block size), whose real, correctness-checked
// behavior at that tile width is also verified here so this file's
// PASS/FAIL doesn't rest on occupancy numbers alone.
//
// PASS requires: every device-property and occupancy API call to succeed;
// the kernel's output to match a CPU reference at tileWidth=16; and active
// blocks per SM to be non-increasing as the swept shared-memory request
// grows (a kernel can never fit *more* resident blocks by asking for more
// shared memory per block). The occupancy percentages themselves are
// printed for information -- they are a property of the specific GPU this
// runs on, not a pass/fail criterion.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

// ---------------------------------------------------------------------------
// Same kernel as 03_dynamic_shared_tiled_matmul.cu -- see that file for the
// full §5.6/Fig. 5.14 rationale for the merged extern __shared__ buffer.
// ---------------------------------------------------------------------------
__global__ void matrixMulTiledDynamicSharedKernel(const float *M, const float *N, float *P,
                                                    int Width, int tileWidth) {
    extern __shared__ float Mds_Nds[];
    float *Mds = Mds_Nds;
    float *Nds = Mds_Nds + tileWidth * tileWidth;

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int Row = by * tileWidth + ty;
    int Col = bx * tileWidth + tx;

    float Pvalue = 0.0f;

    for (int ph = 0; ph < Width / tileWidth; ++ph) {
        Mds[ty * tileWidth + tx] = M[Row * Width + ph * tileWidth + tx];
        Nds[ty * tileWidth + tx] = N[(ph * tileWidth + ty) * Width + Col];

        __syncthreads();

        for (int k = 0; k < tileWidth; ++k) {
            Pvalue += Mds[ty * tileWidth + k] * Nds[k * tileWidth + tx];
        }

        __syncthreads();
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
    bool ok = true;

    cudaDeviceProp devProp;
    CUDA_CHECK(cudaGetDeviceProperties(&devProp, 0));
    printf("Device 0: %s (compute capability %d.%d)\n", devProp.name, devProp.major,
           devProp.minor);
    printf("  multiProcessorCount           : %d\n", devProp.multiProcessorCount);
    printf("  maxThreadsPerMultiProcessor    : %d\n", devProp.maxThreadsPerMultiProcessor);
    printf("  sharedMemPerBlock (default)    : %zu\n", devProp.sharedMemPerBlock);
    printf("  sharedMemPerMultiprocessor     : %zu\n", devProp.sharedMemPerMultiprocessor);
    printf("\n");

    // §5.6's own block size: 256 threads. Held fixed throughout the sweep
    // below so only the shared-memory dimension varies.
    const int blockSize = 256;

    // Hypothetical per-block dynamic shared memory requests, 0 up to the
    // classic 48 KB default per-block limit (no cudaFuncSetAttribute opt-in
    // needed to query or stay within this range).
    const size_t sweepBytes[] = {0,     2048,  4096,  8192,  16384,
                                 24576, 32768, 40960, 49152};

    printf("%12s %16s %10s\n", "smem/block", "activeBlk/SM", "occ%");
    int prevActive = -1;
    for (size_t bytes : sweepBytes) {
        int activeBlocks = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &activeBlocks, matrixMulTiledDynamicSharedKernel, blockSize, bytes));

        double occ =
            100.0 * activeBlocks * blockSize / static_cast<double>(devProp.maxThreadsPerMultiProcessor);
        printf("%10zu B %14d %9.1f%%\n", bytes, activeBlocks, occ);

        // §5.6's lesson, checked against the real API rather than asserted
        // by hand: requesting more shared memory per block can never let
        // *more* blocks fit per SM.
        if (prevActive >= 0 && activeBlocks > prevActive) {
            ok = false;
            fprintf(stderr,
                    "Occupancy sanity check failed at smem=%zu B: activeBlocks (%d) exceeds "
                    "the previous, smaller-smem-request's (%d)\n",
                    bytes, activeBlocks, prevActive);
        }
        prevActive = activeBlocks;
    }
    printf("\n");

    // Real, correctness-checked run at tileWidth=16 (16*16 == 256 threads,
    // matching blockSize above), on a modest array.
    const int Width = 256;
    const int tileWidth = 16;
    size_t count = static_cast<size_t>(Width) * Width;
    size_t size = count * sizeof(float);

    std::vector<float> M_h(count), N_h(count), P_ref(count), P_h(count);
    for (size_t i = 0; i < count; ++i) {
        M_h[i] = static_cast<float>(i % 13) * 0.1f - 0.6f;
        N_h[i] = static_cast<float>(i % 7) * 0.2f - 0.6f;
    }
    matrixMul_h(M_h.data(), N_h.data(), P_ref.data(), Width);

    float *M_d, *N_d, *P_d;
    CUDA_CHECK(cudaMalloc((void **)&M_d, size));
    CUDA_CHECK(cudaMalloc((void **)&N_d, size));
    CUDA_CHECK(cudaMalloc((void **)&P_d, size));
    CUDA_CHECK(cudaMemcpy(M_d, M_h.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(N_d, N_h.data(), size, cudaMemcpyHostToDevice));

    dim3 dimBlock(tileWidth, tileWidth, 1);
    dim3 dimGrid(Width / tileWidth, Width / tileWidth, 1);
    size_t sharedBytes = 2ull * tileWidth * tileWidth * sizeof(float);
    matrixMulTiledDynamicSharedKernel<<<dimGrid, dimBlock, sharedBytes>>>(M_d, N_d, P_d, Width,
                                                                           tileWidth);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(P_h.data(), P_d, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(M_d));
    CUDA_CHECK(cudaFree(N_d));
    CUDA_CHECK(cudaFree(P_d));

    for (size_t i = 0; i < count; ++i) {
        if (!nearlyEqual(P_h[i], P_ref[i])) {
            ok = false;
            fprintf(stderr, "Mismatch at i=%zu: gpu=%f cpu=%f\n", i, P_h[i], P_ref[i]);
            break;
        }
    }

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
