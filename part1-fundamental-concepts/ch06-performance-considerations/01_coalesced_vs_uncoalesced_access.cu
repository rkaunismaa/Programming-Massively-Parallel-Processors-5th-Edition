// Chapter 6: Performance considerations
// §6.1  Global memory access coalescing -- Fig. 6.2 (coalesced) vs Fig. 6.3
// (un-coalesced) access patterns.
//
// The book's example: a matrix N is used as the second input to a matrix
// multiplication. Fig. 6.2 shows that when N is stored row-major, the index
// expression k*Width+col makes consecutive threads (consecutive col) access
// consecutive memory locations -- coalesced. Fig. 6.3 shows that when N is
// instead stored column-major (equivalently: accessing the transpose of a
// row-major matrix), the index expression col*Width+k makes consecutive
// threads access locations Width elements apart -- un-coalesced.
//
// This file isolates exactly that access-pattern contrast from the matmul
// context: both kernels compute the same logical result -- the column sums
// of a Width x Width matrix N -- but read the values from two different
// flat buffers holding the same logical matrix in different physical
// layouts:
//   - N_row: row-major, N_row[r*Width+c] == N[r][c]           (Fig. 6.2 layout)
//   - N_col: column-major, N_col[c*Width+r] == N[r][c]        (Fig. 6.3 layout)
// The coalesced kernel reads N_row with index k*Width+col (Fig. 6.2's exact
// expression). The uncoalesced kernel reads N_col with index col*Width+k
// (Fig. 6.3's exact expression) -- same logical values, same amount of
// work, only the memory layout/access-pattern differs.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

// Logical matrix element N[r][c], used to fill both physical layouts and to
// compute the CPU reference.
static inline float logicalN(int r, int c) {
    return static_cast<float>((r * 31 + c * 7) % 101) * 0.01f - 0.5f;
}

// ---------------------------------------------------------------------------
// §6.1, Fig. 6.2: coalesced access. Consecutive threads (consecutive col)
// read consecutive elements of N_row for each fixed k -- the accesses of a
// warp in iteration k are adjacent in memory.
// ---------------------------------------------------------------------------
__global__ void colSumCoalescedKernel(const float *N_row, float *colSum, int Width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < Width) {
        float sum = 0.0f;
        for (int k = 0; k < Width; ++k) {
            sum += N_row[k * Width + col];
        }
        colSum[col] = sum;
    }
}

// ---------------------------------------------------------------------------
// §6.1, Fig. 6.3: un-coalesced access. Consecutive threads (consecutive col)
// read elements of N_col that are Width apart for each fixed k -- the
// accesses of a warp in iteration k are scattered across Width-sized strides.
// ---------------------------------------------------------------------------
__global__ void colSumUncoalescedKernel(const float *N_col, float *colSum, int Width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < Width) {
        float sum = 0.0f;
        for (int k = 0; k < Width; ++k) {
            sum += N_col[col * Width + k];
        }
        colSum[col] = sum;
    }
}

int main() {
    const int Width = 4096;

    size_t count = static_cast<size_t>(Width) * Width;
    size_t matSize = count * sizeof(float);
    size_t vecSize = static_cast<size_t>(Width) * sizeof(float);

    std::vector<float> N_row_h(count), N_col_h(count);
    std::vector<float> colSum_ref(Width);

    for (int r = 0; r < Width; ++r) {
        for (int c = 0; c < Width; ++c) {
            float v = logicalN(r, c);
            N_row_h[static_cast<size_t>(r) * Width + c] = v;         // row-major: (r,c) -> r*Width+c
            N_col_h[static_cast<size_t>(c) * Width + r] = v;         // column-major: (r,c) -> c*Width+r
        }
    }

    printf("Computing CPU reference (Width=%d)...\n", Width);
    for (int col = 0; col < Width; ++col) {
        float sum = 0.0f;
        for (int k = 0; k < Width; ++k) {
            sum += logicalN(k, col);
        }
        colSum_ref[col] = sum;
    }

    float *N_row_d, *N_col_d, *colSum_d;
    CUDA_CHECK(cudaMalloc((void **)&N_row_d, matSize));
    CUDA_CHECK(cudaMalloc((void **)&N_col_d, matSize));
    CUDA_CHECK(cudaMalloc((void **)&colSum_d, vecSize));
    CUDA_CHECK(cudaMemcpy(N_row_d, N_row_h.data(), matSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(N_col_d, N_col_h.data(), matSize, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (Width + blockSize - 1) / blockSize;

    // Warm up both kernels once each (discarded) before timing, so one-time
    // PTX->SASS JIT cost doesn't get attributed to whichever kernel happens
    // to launch first.
    colSumCoalescedKernel<<<gridSize, blockSize>>>(N_row_d, colSum_d, Width);
    CUDA_CHECK(cudaGetLastError());
    colSumUncoalescedKernel<<<gridSize, blockSize>>>(N_col_d, colSum_d, Width);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> colSum_coalesced_h(Width), colSum_uncoalesced_h(Width);

    GpuTimer timer;
    timer.start();
    colSumCoalescedKernel<<<gridSize, blockSize>>>(N_row_d, colSum_d, Width);
    CUDA_CHECK(cudaGetLastError());
    float coalesced_ms = timer.stopAndGetMs();
    CUDA_CHECK(cudaMemcpy(colSum_coalesced_h.data(), colSum_d, vecSize, cudaMemcpyDeviceToHost));

    timer.start();
    colSumUncoalescedKernel<<<gridSize, blockSize>>>(N_col_d, colSum_d, Width);
    CUDA_CHECK(cudaGetLastError());
    float uncoalesced_ms = timer.stopAndGetMs();
    CUDA_CHECK(cudaMemcpy(colSum_uncoalesced_h.data(), colSum_d, vecSize, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(N_row_d));
    CUDA_CHECK(cudaFree(N_col_d));
    CUDA_CHECK(cudaFree(colSum_d));

    bool coalesced_ok = true, uncoalesced_ok = true;
    for (int col = 0; col < Width; ++col) {
        if (coalesced_ok && !nearlyEqual(colSum_coalesced_h[col], colSum_ref[col])) {
            coalesced_ok = false;
            fprintf(stderr, "Coalesced mismatch at col=%d: gpu=%f cpu=%f\n",
                    col, colSum_coalesced_h[col], colSum_ref[col]);
        }
        if (uncoalesced_ok && !nearlyEqual(colSum_uncoalesced_h[col], colSum_ref[col])) {
            uncoalesced_ok = false;
            fprintf(stderr, "Uncoalesced mismatch at col=%d: gpu=%f cpu=%f\n",
                    col, colSum_uncoalesced_h[col], colSum_ref[col]);
        }
    }

    printf("Width = %d, %d threads (1 per column)\n", Width, Width);
    printf("Coalesced   (§6.1, Fig. 6.2, k*Width+col)   kernel time: %.3f ms  [%s]\n",
           coalesced_ms, coalesced_ok ? "match" : "MISMATCH");
    printf("Uncoalesced (§6.1, Fig. 6.3, col*Width+k)   kernel time: %.3f ms  [%s]\n",
           uncoalesced_ms, uncoalesced_ok ? "match" : "MISMATCH");
    printf("Slowdown (uncoalesced/coalesced): %.2fx\n", uncoalesced_ms / coalesced_ms);

    bool ok = coalesced_ok && uncoalesced_ok;
    printf("%s\n", ok ? "PASS" : "FAIL");

    return ok ? 0 : 1;
}
