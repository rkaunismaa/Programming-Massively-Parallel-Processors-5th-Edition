// Chapter 23: Multi-GPU programming
// Section 23.2 "Multi-GPU stencil with MPI" (Fig. 23.5-23.11).
//
// NOT COMPILED/RUN ON THIS MACHINE: no system-wide MPI development package
// is installed here (no <mpi.h>). This file is a full, real implementation
// written to match §23.2's description and its (image-only, hence prose-
// reconstructed) code figures faithfully -- see the per-block comments
// below citing exactly which paragraph/figure each piece comes from. It is
// not a stub: every MPI call below is a real MPI-3 call with the standard
// signature, and the domain-decomposition/halo-exchange logic is modeled
// directly on 01_stencil_singlegpu_baseline.cu's Jacobi kernel and L2-norm
// reduction, per the task's requirement that each rank compute the same
// interior-point stencil as file 01, just on its own slab.
//
// ---------------------------------------------------------------------
// Domain decomposition (§23.1, Fig. 23.2-23.3):
// ---------------------------------------------------------------------
// The book partitions the 2D grid across GPUs along the y-dimension only
// (row-major layout => each partition's rows are contiguous in memory,
// simplifying the halo copy). Every rank owns the full row width `nx` and
// a contiguous band of `nyLocalInterior` rows, plus one halo row above and
// one halo row below (`nyLocal = nyLocalInterior + 2`). x=0 and x=nx-1
// remain genuine, non-decomposed Dirichlet boundaries in every rank
// (never overwritten), exactly like the single-GPU baseline's left/right
// edges.
//
// One deliberate difference from 01_stencil_singlegpu_baseline.cu: per
// §23.2's own text, the top/bottom *rank* neighbors are computed with a
// wrap-around (periodic) rule specifically so that no rank needs special-
// casing for "I have no neighbor above/below me":
//
//   "Rather than simply computing rank - 1 and rank + 1, the code uses a
//    wrap-around strategy where the topmost rank treats the bottommost
//    rank as its top neighbor, and vice versa. This wrap-around strategy
//    is often referred to as the periodic boundary condition technique."
//
// This means the y-dimension of the *global* multi-rank domain is
// effectively periodic (a torus) in this and all following multi-GPU
// files, whereas the single-GPU baseline (file 01, no ranks at all) has a
// true fixed Dirichlet top/bottom edge. This is a real difference in the
// modeled problem's boundary conditions between file 01 and files 02-05,
// grounded directly in the paragraph quoted above -- not an inconsistency.
// x=0/x=nx-1 stay true, non-periodic Dirichlet boundaries in every file
// because the book never decomposes the x-dimension.

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#include "../../common/cuda_utils.h"

// ---------------------------------------------------------------------
// Same generalized Jacobi kernel as 01_stencil_singlegpu_baseline.cu:
// computes rows [1, numRows-2] of a `numRows`-row span, treating row 0 and
// row numRows-1 of the given span as fixed input (boundary/halo) rows.
// Called here with numRows = nyLocal (the whole local slab in one launch),
// which is exactly Fig. 23.1's kernel applied to one rank's partition.
// ---------------------------------------------------------------------
__global__ void jacobiKernel(float* out, const float* in, int nx, int numRows,
                              float* l2normSq) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    float mySq = 0.0f;
    if (ix > 0 && ix < nx - 1 && iy > 0 && iy < numRows - 1) {
        int idx = iy * nx + ix;
        float newVal = 0.25f * (in[idx - 1] + in[idx + 1] + in[idx - nx] + in[idx + nx]);
        out[idx] = newVal;
        float residue = newVal - in[idx];
        mySq = residue * residue;
    }

    extern __shared__ float sdata[];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    sdata[tid] = mySq;
    __syncthreads();
    for (int stride = (blockDim.x * blockDim.y) / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(l2normSq, sdata[0]);
}

static void launchJacobiKernel(float* out, const float* in, int nx, int numRows,
                                float* l2normSq, cudaStream_t stream = 0) {
    dim3 block(16, 16);
    dim3 grid((nx + block.x - 1) / block.x, (numRows + block.y - 1) / block.y);
    size_t shmem = static_cast<size_t>(block.x) * block.y * sizeof(float);
    jacobiKernel<<<grid, block, shmem, stream>>>(out, in, nx, numRows, l2normSq);
}

__global__ void initGridKernel(float* grid, int nx, int nyLocal) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < nx && iy < nyLocal) {
        // x=0/x=nx-1 held at 0 (true, non-decomposed Dirichlet edges);
        // everything else initialized to 0. Since the multi-rank domain is
        // periodic in y (see file header), there is no single "hot" edge
        // as in file 01 -- this is purely an illustrative initial
        // condition, never exercised on this machine.
        grid[iy * nx + ix] = 0.0f;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    // One GPU per rank (assumes numRanks <= number of visible GPUs on the
    // node, or a launcher that has already restricted CUDA_VISIBLE_DEVICES
    // per rank -- both are standard multi-GPU-per-node MPI deployment
    // patterns; the book does not dictate one over the other).
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));

    // Global grid: nx columns, nyTotal rows, decomposed along y only.
    const int nx = 130;
    const int nyTotalInterior = 512;  // must be divisible by numRanks
    const int nyLocalInterior = nyTotalInterior / numRanks;
    const int nyLocal = nyLocalInterior + 2;  // + top halo row + bottom halo row

    const int maxIters = 2000;
    const float tol = 1e-6f;

    size_t bytes = static_cast<size_t>(nx) * nyLocal * sizeof(float);

    float *d_input = nullptr, *d_output = nullptr, *d_l2normSq = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMalloc(&d_l2normSq, sizeof(float)));

    dim3 initBlock(16, 16);
    dim3 initGrid((nx + initBlock.x - 1) / initBlock.x, (nyLocal + initBlock.y - 1) / initBlock.y);
    initGridKernel<<<initGrid, initBlock>>>(d_input, nx, nyLocal);
    initGridKernel<<<initGrid, initBlock>>>(d_output, nx, nyLocal);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // §23.2: "the code uses a wrap-around strategy where the topmost rank
    // treats the bottommost rank as its top neighbor, and vice versa"
    // (periodic boundary condition technique -- see file header).
    int topNeighbor = (rank == 0) ? (numRanks - 1) : (rank - 1);
    int bottomNeighbor = (rank == numRanks - 1) ? 0 : (rank + 1);

    int iter = 0;
    float l2norm = 1e30f;
    while (iter < maxIters && l2norm > tol) {
        // Fig. 23.7 line 04: reset the L2 norm accumulator to 0.
        CUDA_CHECK(cudaMemset(d_l2normSq, 0, sizeof(float)));

        // Fig. 23.7 line 07-08: launch the Jacobi kernel for this rank's
        // whole local slab and wait for it to finish before halo exchange.
        launchJacobiKernel(d_output, d_input, nx, nyLocal, d_l2normSq);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Fig. 23.7 lines 11-18, using MPI_Sendrecv (Fig. 23.10) for the
        // halo exchange. Both pointers passed to MPI_Sendrecv are GPU
        // device pointers; this relies on CUDA-aware MPI (§23.2, "Passing
        // GPU device memory pointers to MPI functions is possible with
        // CUDA-aware MPI... Most MPI implementations (e.g., MPICH,
        // OpenMPI, MVAPICH2) are designed to be CUDA-aware").
        //
        // First MPI_Sendrecv: send my new top boundary row (row 1) to my
        // top neighbor; receive my bottom halo row (row nyLocal-1) from my
        // bottom neighbor.
        MPI_Sendrecv(d_output + 1 * nx, nx, MPI_FLOAT, topNeighbor, 0,
                     d_output + (nyLocal - 1) * nx, nx, MPI_FLOAT, bottomNeighbor, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Second MPI_Sendrecv: send my new bottom boundary row
        // (row nyLocal-2) to my bottom neighbor; receive my top halo row
        // (row 0) from my top neighbor.
        MPI_Sendrecv(d_output + (nyLocal - 2) * nx, nx, MPI_FLOAT, bottomNeighbor, 1,
                     d_output, nx, MPI_FLOAT, topNeighbor, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Fig. 23.7 lines 20-25, Fig. 23.11: copy this rank's partial L2
        // norm square to the host, then MPI_Allreduce (MPI_SUM) across all
        // ranks, then take the square root on the host. Copied to the host
        // first because the result is needed on the host to check
        // convergence anyway (§23.2 closing paragraph).
        float l2normSq_h = 0.0f;
        CUDA_CHECK(cudaMemcpy(&l2normSq_h, d_l2normSq, sizeof(float), cudaMemcpyDeviceToHost));
        float l2normSumSq_h = 0.0f;
        MPI_Allreduce(&l2normSq_h, &l2normSumSq_h, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        l2norm = std::sqrt(l2normSumSq_h);

        // Fig. 23.7 line 27: double-buffer swap.
        std::swap(d_input, d_output);
        ++iter;

        if (rank == 0 && (iter % 500 == 0)) {
            printf("iter %d: l2norm = %.6e\n", iter, l2norm);
        }
    }

    if (rank == 0) {
        printf("Rank 0: finished after %d iterations, final L2 norm = %.6e\n", iter, l2norm);
        printf("(not executed on this machine -- see README)\n");
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_l2normSq));

    MPI_Finalize();
    return 0;
}
