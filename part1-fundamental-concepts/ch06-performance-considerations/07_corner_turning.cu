// Chapter 6: Performance considerations
// §6.1  Global memory access coalescing -- corner turning (Fig. 6.4)
// §6.4  Shared memory bank conflicts -- the corner-turning example, revisited
//
// §6.1 introduces *corner turning* for the case where a tiled matmul's second
// input N is stored column-major (e.g. because it's really the transpose of
// a row-major matrix accessed in place -- see 01_coalesced_vs_uncoalesced_access.cu
// for that same scenario without tiling). Loading N's tile the same way M's
// tile is loaded gives every thread its "own" element, but because N is
// column-major this makes consecutive threads (consecutive tx, fixed ty)
// read locations Width apart -- uncoalesced (Fig. 6.4(a)). The fix: "exchange
// the roles of threadIdx.x and threadIdx.y when each thread calculates the
// linearized index for loading the N input tile" (Fig. 6.4(b)) -- thread
// (tx,ty) loads the element that would "naturally" belong to thread (ty,tx),
// so consecutive tx now advances N's contiguous (row) dimension: coalesced.
//
// §6.4 then revisits this *exact* example: "consider the corner turning
// optimization in Fig. 6.4(b) ... Although the threads load from global
// memory in a coalesced manner, their stores to shared memory have a strided
// pattern" -- storing the corner-turned load at Nds[tx][ty] (needed so the
// later dot-product loop can still read Nds[k][tx] unchanged) means a warp
// (fixed ty, tx = 0..31) writes to linear shared-memory offsets tx*TILE_DIM+ty,
// all reducing to the same bank mod TILE_DIM -- the section's own 32-way
// conflict example, arising here rather than in a standalone toy kernel.
// Padding to Nds[TILE_WIDTH][TILE_WIDTH+1] (03_shared_memory_bank_conflicts.cu's
// fix) removes it.
//
// This file implements all three stages as three complete tiled-matmul
// kernels multiplying a row-major M by a *column-major* N, all checked
// against the same CPU reference:
//   1. matrixMulNoCornerTurningKernel:      uncoalesced global load,
//                                           conflict-free shared store.
//   2. matrixMulCornerTurningKernel:        coalesced global load,
//                                           conflicting shared store.
//   3. matrixMulCornerTurningPaddedKernel:  coalesced global load,
//                                           conflict-free shared store.
// Only the single line that loads a tile of N differs between the three;
// M's loading, the dot-product loop, and the P write are identical.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

#define TILE_WIDTH 32

// Deterministic per-logical-element values, independent of physical layout.
static inline float valM(int row, int col) {
    return static_cast<float>((row * 13 + col * 3) % 97) * 0.01f - 0.48f;
}
static inline float valN(int row, int col) {
    return static_cast<float>((row * 7 + col * 31) % 101) * 0.01f - 0.5f;
}

// ---------------------------------------------------------------------------
// §6.1, Fig. 6.4(a): no corner turning. N is column-major
// (N_col[col*Width+row] == N[row][col]), but thread (tx,ty) loads the
// element at its "own" (row=ph*TILE_WIDTH+ty, col=Col) -- for a warp (tx
// varying, ty fixed), consecutive threads' indices are Width apart.
// ---------------------------------------------------------------------------
__global__ void matrixMulNoCornerTurningKernel(const float *M, const float *N_col, float *P,
                                                 int Width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0f;
    for (int ph = 0; ph < Width / TILE_WIDTH; ++ph) {
        Mds[ty][tx] = M[Row * Width + ph * TILE_WIDTH + tx];
        Nds[ty][tx] = N_col[Col * Width + (ph * TILE_WIDTH + ty)];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    P[Row * Width + Col] = Pvalue;
}

// ---------------------------------------------------------------------------
// §6.1, Fig. 6.4(b): corner turning. Thread (tx,ty) loads N's element at
// (row=ph*TILE_WIDTH+tx, col=bx*TILE_WIDTH+ty) instead -- for a warp (tx
// varying, ty fixed), consecutive threads' indices are 1 apart (row is
// N_col's contiguous dimension): coalesced. Storing at Nds[tx][ty] (so the
// dot-product loop below, unchanged, still reads the right value from
// Nds[k][tx]) makes a warp's store hit linear offsets tx*TILE_WIDTH+ty --
// all the same bank mod TILE_WIDTH: §6.4's 32-way conflict, unfixed here.
// ---------------------------------------------------------------------------
__global__ void matrixMulCornerTurningKernel(const float *M, const float *N_col, float *P,
                                               int Width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];  // unpadded: bank-conflicted store below

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0f;
    for (int ph = 0; ph < Width / TILE_WIDTH; ++ph) {
        Mds[ty][tx] = M[Row * Width + ph * TILE_WIDTH + tx];
        Nds[tx][ty] = N_col[(bx * TILE_WIDTH + ty) * Width + (ph * TILE_WIDTH + tx)];
        __syncthreads();

        // Unchanged from the standard row-major-N tiled kernel: Nds[k][tx]
        // already holds N[ph*TILE_WIDTH+k][Col], because of how the load
        // above placed it.
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    P[Row * Width + Col] = Pvalue;
}

// ---------------------------------------------------------------------------
// §6.1 corner turning + §6.4 padding: identical to the kernel above except
// Nds gains one extra column, shifting a warp's store to linear offsets
// tx*(TILE_WIDTH+1)+ty -- consecutive tx now land in consecutive banks
// (mod TILE_WIDTH): conflict-free. Coalesced global load *and*
// conflict-free shared store.
// ---------------------------------------------------------------------------
__global__ void matrixMulCornerTurningPaddedKernel(const float *M, const float *N_col, float *P,
                                                     int Width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH + 1];  // padded: conflict-free store below

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0f;
    for (int ph = 0; ph < Width / TILE_WIDTH; ++ph) {
        Mds[ty][tx] = M[Row * Width + ph * TILE_WIDTH + tx];
        Nds[tx][ty] = N_col[(bx * TILE_WIDTH + ty) * Width + (ph * TILE_WIDTH + tx)];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    P[Row * Width + Col] = Pvalue;
}

// CPU reference: identical dot-product formula, independent of either
// input's physical storage layout.
void matrixMul_h(const float *M, const float *N_col, float *P, int Width) {
    (void)M;
    (void)N_col;
    for (int row = 0; row < Width; ++row) {
        for (int col = 0; col < Width; ++col) {
            float Pvalue = 0.0f;
            for (int k = 0; k < Width; ++k) {
                Pvalue += valM(row, k) * valN(k, col);
            }
            P[row * Width + col] = Pvalue;
        }
    }
}

int main() {
    // Matmul is O(Width^3) on the CPU reference, unlike this chapter's
    // pure-memory-access samples (01, 03) which can afford Width=4096;
    // 1024 matches the other tiled-matmul samples in this repo (Ch. 5's
    // 01_tiled_matrix_multiplication.cu, this chapter's 04/05/06).
    const int Width = 1024;

    if (Width % TILE_WIDTH != 0) {
        fprintf(stderr, "Width must be a multiple of TILE_WIDTH for this file\n");
        return 1;
    }

    size_t count = static_cast<size_t>(Width) * Width;
    size_t size = count * sizeof(float);

    std::vector<float> M_h(count), N_col_h(count), P_ref(count);
    for (int row = 0; row < Width; ++row) {
        for (int col = 0; col < Width; ++col) {
            M_h[static_cast<size_t>(row) * Width + col] = valM(row, col);
            // Column-major: N_col[col*Width+row] == N[row][col].
            N_col_h[static_cast<size_t>(col) * Width + row] = valN(row, col);
        }
    }

    printf("Computing CPU reference (Width=%d, %zu elements)...\n", Width, count);
    matrixMul_h(M_h.data(), N_col_h.data(), P_ref.data(), Width);

    float *M_d, *N_d, *P_d;
    CUDA_CHECK(cudaMalloc((void **)&M_d, size));
    CUDA_CHECK(cudaMalloc((void **)&N_d, size));
    CUDA_CHECK(cudaMalloc((void **)&P_d, size));
    CUDA_CHECK(cudaMemcpy(M_d, M_h.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(N_d, N_col_h.data(), size, cudaMemcpyHostToDevice));

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid(Width / TILE_WIDTH, Width / TILE_WIDTH, 1);

    // Warm up all three kernels once each (discarded) before timing.
    matrixMulNoCornerTurningKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, Width);
    CUDA_CHECK(cudaGetLastError());
    matrixMulCornerTurningKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, Width);
    CUDA_CHECK(cudaGetLastError());
    matrixMulCornerTurningPaddedKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, Width);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> P_noct_h(count), P_ct_h(count), P_ctp_h(count);
    GpuTimer timer;

    timer.start();
    matrixMulNoCornerTurningKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, Width);
    CUDA_CHECK(cudaGetLastError());
    float noct_ms = timer.stopAndGetMs();
    CUDA_CHECK(cudaMemcpy(P_noct_h.data(), P_d, size, cudaMemcpyDeviceToHost));

    timer.start();
    matrixMulCornerTurningKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, Width);
    CUDA_CHECK(cudaGetLastError());
    float ct_ms = timer.stopAndGetMs();
    CUDA_CHECK(cudaMemcpy(P_ct_h.data(), P_d, size, cudaMemcpyDeviceToHost));

    timer.start();
    matrixMulCornerTurningPaddedKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, Width);
    CUDA_CHECK(cudaGetLastError());
    float ctp_ms = timer.stopAndGetMs();
    CUDA_CHECK(cudaMemcpy(P_ctp_h.data(), P_d, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(M_d));
    CUDA_CHECK(cudaFree(N_d));
    CUDA_CHECK(cudaFree(P_d));

    bool noct_ok = true, ct_ok = true, ctp_ok = true;
    for (size_t i = 0; i < count; ++i) {
        if (noct_ok && !nearlyEqual(P_noct_h[i], P_ref[i])) {
            noct_ok = false;
            fprintf(stderr, "No-corner-turning mismatch at i=%zu: gpu=%f cpu=%f\n", i,
                    P_noct_h[i], P_ref[i]);
        }
        if (ct_ok && !nearlyEqual(P_ct_h[i], P_ref[i])) {
            ct_ok = false;
            fprintf(stderr, "Corner-turning mismatch at i=%zu: gpu=%f cpu=%f\n", i, P_ct_h[i],
                    P_ref[i]);
        }
        if (ctp_ok && !nearlyEqual(P_ctp_h[i], P_ref[i])) {
            ctp_ok = false;
            fprintf(stderr, "Corner-turning-padded mismatch at i=%zu: gpu=%f cpu=%f\n", i,
                    P_ctp_h[i], P_ref[i]);
        }
    }

    printf("Width = %d, TILE_WIDTH = %d, N stored column-major\n", Width, TILE_WIDTH);
    printf("No corner turning       (§6.1, Fig. 6.4a: uncoalesced load,  conflict-free store) : "
           "%.3f ms  [%s]\n",
           noct_ms, noct_ok ? "match" : "MISMATCH");
    printf("Corner turning          (§6.1, Fig. 6.4b: coalesced load,    conflicted store)    : "
           "%.3f ms  [%s]\n",
           ct_ms, ct_ok ? "match" : "MISMATCH");
    printf("Corner turning + padded (§6.1 + §6.4:      coalesced load,    conflict-free store) : "
           "%.3f ms  [%s]\n",
           ctp_ms, ctp_ok ? "match" : "MISMATCH");
    printf("Speedup, no-CT -> CT+padded: %.2fx\n", noct_ms / ctp_ms);

    bool ok = noct_ok && ct_ok && ctp_ok;
    printf("%s\n", ok ? "PASS" : "FAIL");

    return ok ? 0 : 1;
}
