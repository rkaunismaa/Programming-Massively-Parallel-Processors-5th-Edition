// Chapter 23: Multi-GPU programming
// Section 23.3 "Overlapping computation and communication" (Fig. 23.12-23.15).
//
// NOT COMPILED/RUN ON THIS MACHINE (see 02_stencil_multigpu_mpi.cu's header
// for why). This file builds directly on 02_stencil_multigpu_mpi.cu: same
// domain decomposition (1D along y, periodic/wrap-around rank topology per
// §23.2), same Jacobi math, same MPI_Sendrecv/MPI_Allreduce halo-exchange
// and L2-norm-reduction calls -- the only change is *how* the single
// Jacobi kernel launch of file 02 is split into three launches on three
// CUDA streams so that communication of the boundary rows overlaps with
// computation of the internal rows, exactly as §23.3 describes:
//
//   "During the first stage (Stage 1), each process calculates its
//    boundary rows that are needed as halo cells by its neighbors in the
//    next iteration... During the second stage (Stage 2), each process
//    performs two activities in parallel. The first is to communicate its
//    new boundary values to its neighbor processes. The second activity is
//    to calculate the rest of the data in the partition."  (§23.3)
//
// ---------------------------------------------------------------------
// How the single kernel call is split into three (§23.3, Fig. 23.14):
// ---------------------------------------------------------------------
// The *same* generalized jacobiKernel(out, in, nx, numRows, l2normSq) from
// file 01/02 is launched three times with different pointer offsets and
// `numRows` spans -- it is not a different kernel, just called on
// sub-views of the local slab:
//
//   - Top boundary row:    launchJacobiKernel(output,            input,            nx, 3,         ..., topStream)
//                           computes only row 1 (global row 1) of the local slab.
//   - Bottom boundary row: launchJacobiKernel(output+(nyLocal-3)*nx, input+(nyLocal-3)*nx, nx, 3, ..., bottomStream)
//                           computes only row nyLocal-2.
//   - Internal rows:       launchJacobiKernel(output+nx,        input+nx,         nx, nyLocal-2, ..., internalStream)
//                           computes rows 2 .. nyLocal-3 (strictly interior, no boundary rows).
//
// Together these three calls compute exactly rows 1..nyLocal-2 with no
// overlap and no gap -- the same total set of points file 02's single
// launch computes, per §23.3: "we can split the kernel call into three
// different calls: one for the top boundary row, one for the bottom
// boundary row, and one for the internal elements."

#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#include "../../common/cuda_utils.h"

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
                                float* l2normSq, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((nx + block.x - 1) / block.x, (numRows + block.y - 1) / block.y);
    size_t shmem = static_cast<size_t>(block.x) * block.y * sizeof(float);
    jacobiKernel<<<grid, block, shmem, stream>>>(out, in, nx, numRows, l2normSq);
}

__global__ void initGridKernel(float* grid, int nx, int nyLocal) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < nx && iy < nyLocal) {
        // The y-topology across ranks is periodic (wrap-around, per §23.2),
        // so there is no true global top/bottom Dirichlet edge here, unlike
        // file 01. x=0/x=nx-1 remain true non-periodic edges in every file
        // since the book never decomposes along x, so we drive the problem
        // from a Dirichlet edge at x=0 instead (analogous to file 01's
        // y=0 edge), matching file 01's ny==0-edge value of 1.0f.
        grid[iy * nx + ix] = (ix == 0) ? 1.0f : 0.0f;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));

    const int nx = 130;
    const int nyTotalInterior = 512;
    const int nyLocalInterior = nyTotalInterior / numRanks;
    const int nyLocal = nyLocalInterior + 2;

    const int maxIters = 2000;
    const float tol = 1e-6f;

    size_t bytes = static_cast<size_t>(nx) * nyLocal * sizeof(float);

    float *d_input = nullptr, *d_output = nullptr, *d_l2normSq = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMalloc(&d_l2normSq, sizeof(float)));

    // l2norm_h must be pinned (page-locked) host memory because it is the
    // target of an asynchronous cudaMemcpyAsync below (§23.3, end of
    // section: "since the host memory pointed to by l2norm_h is involved
    // in asynchronous memory operations such as cudaMemcpyAsync, it cannot
    // be allocated using the standard host memory allocation methods such
    // as malloc(). Instead, it must be allocated in pinned memory... using
    // the cudaMallocHost function").
    float* l2norm_h = nullptr;
    CUDA_CHECK(cudaMallocHost(&l2norm_h, sizeof(float)));

    dim3 initBlock(16, 16);
    dim3 initGrid((nx + initBlock.x - 1) / initBlock.x, (nyLocal + initBlock.y - 1) / initBlock.y);
    initGridKernel<<<initGrid, initBlock>>>(d_input, nx, nyLocal);
    initGridKernel<<<initGrid, initBlock>>>(d_output, nx, nyLocal);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int topNeighbor = (rank == 0) ? (numRanks - 1) : (rank - 1);
    int bottomNeighbor = (rank == numRanks - 1) ? 0 : (rank + 1);

    // Three streams: topStream/bottomStream carry the boundary-row kernels
    // and must run ahead of internalStream so their halo data is ready to
    // send as early as possible (§23.3: "priority should be given to the
    // boundary grid points in order to initiate the halo exchange as soon
    // as possible"). Created with explicit priorities so the scheduler
    // does not let the (much larger) internal kernel starve them:
    //
    //   int low, high;
    //   cudaDeviceGetStreamPriorityRange(&low, &high);
    //   cudaStreamCreateWithPriority(&internalStream, cudaStreamDefault, low);
    //   cudaStreamCreateWithPriority(&topStream, cudaStreamDefault, high);
    //   cudaStreamCreateWithPriority(&bottomStream, cudaStreamDefault, high);
    int lowPriority = 0, highPriority = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lowPriority, &highPriority));
    cudaStream_t topStream, bottomStream, internalStream;
    CUDA_CHECK(cudaStreamCreateWithPriority(&internalStream, cudaStreamDefault, lowPriority));
    CUDA_CHECK(cudaStreamCreateWithPriority(&topStream, cudaStreamDefault, highPriority));
    CUDA_CHECK(cudaStreamCreateWithPriority(&bottomStream, cudaStreamDefault, highPriority));

    // Events (§23.3): `resetL2` marks completion of the L2-norm reset so
    // the boundary-row kernels (in topStream/bottomStream) don't race
    // ahead of it; `topDone`/`bottomDone` mark completion of the boundary
    // kernels so internalStream's final cudaMemcpyAsync of the L2 norm
    // (which needs contributions from *all three* kernels) waits for them.
    // cudaEventDisableTiming because these events are used purely for
    // stream-to-stream ordering, not timing.
    cudaEvent_t resetL2, topDone, bottomDone;
    CUDA_CHECK(cudaEventCreateWithFlags(&resetL2, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&topDone, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&bottomDone, cudaEventDisableTiming));

    int iter = 0;
    float l2norm = 1e30f;
    while (iter < maxIters && l2norm > tol) {
        // Fig. 23.14 line 04: non-blocking reset of the L2 norm accumulator
        // in internalStream, followed by recording the resetL2 event.
        CUDA_CHECK(cudaMemsetAsync(d_l2normSq, 0, sizeof(float), internalStream));
        CUDA_CHECK(cudaEventRecord(resetL2, internalStream));

        // Fig. 23.14 lines 06-07: topStream/bottomStream wait for the L2
        // norm reset before their kernels run.
        CUDA_CHECK(cudaStreamWaitEvent(topStream, resetL2, 0));
        CUDA_CHECK(cudaStreamWaitEvent(bottomStream, resetL2, 0));

        // Fig. 23.14 line 10: top boundary row -- needs the first three
        // rows (halo, boundary, first-internal) as input; computes row 1.
        launchJacobiKernel(d_output, d_input, nx, 3, d_l2normSq, topStream);
        CUDA_CHECK(cudaEventRecord(topDone, topStream));

        // Fig. 23.14 lines 12-13: bottom boundary row -- needs the last
        // three rows as input; computes row nyLocal-2.
        launchJacobiKernel(d_output + (nyLocal - 3) * nx, d_input + (nyLocal - 3) * nx,
                            nx, 3, d_l2normSq, bottomStream);
        CUDA_CHECK(cudaEventRecord(bottomDone, bottomStream));

        // Fig. 23.14 lines 15-16: internal rows -- skip the first halo
        // row; span of nyLocal-2 rows means the kernel's own boundary
        // check computes rows 2..nyLocal-3 relative to the full slab.
        launchJacobiKernel(d_output + nx, d_input + nx, nx, nyLocal - 2, d_l2normSq, internalStream);

        // Fig. 23.14 lines 19-20: internalStream waits for both boundary
        // kernels before copying the (now complete) L2 norm sum to the
        // host, so the copy reflects all three kernels' contributions.
        CUDA_CHECK(cudaStreamWaitEvent(internalStream, topDone, 0));
        CUDA_CHECK(cudaStreamWaitEvent(internalStream, bottomDone, 0));

        // Fig. 23.14 line 21: non-blocking copy of the L2 norm square to
        // pinned host memory, inserted into internalStream *before* the
        // MPI_Sendrecv calls below. Per §23.3: "One further optimization
        // applied to our host code..., based on the fact that
        // cudaMemcpyAsync is non-blocking, is that we can use
        // cudaMemcpyAsync to insert the memory copy of the l2norm from
        // the global memory to the host memory before the calls to
        // MPI_Sendrecv... Doing so overlaps the memory copy from the GPU
        // to the host CPU with the network communication performing the
        // halo exchange if the kernel computing the internal grid points
        // finishes before the halo exchange does." I.e. the overlap is
        // between this D2H copy (once its stream prerequisites are met)
        // and the two host-blocking MPI_Sendrecv calls below, not with
        // any halo exchange that has already happened -- there is none
        // yet at this point.
        CUDA_CHECK(cudaMemcpyAsync(l2norm_h, d_l2normSq, sizeof(float), cudaMemcpyDeviceToHost, internalStream));

        // Fig. 23.14 line 27: host waits for the top-boundary kernel to
        // finish before initiating its halo exchange.
        CUDA_CHECK(cudaStreamSynchronize(topStream));
        MPI_Sendrecv(d_output + 1 * nx, nx, MPI_FLOAT, topNeighbor, 0,
                     d_output + (nyLocal - 1) * nx, nx, MPI_FLOAT, bottomNeighbor, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Fig. 23.14 line 32: host waits for the bottom-boundary kernel to
        // finish before initiating its halo exchange. Meanwhile the
        // internal kernel (in internalStream) and the cudaMemcpyAsync
        // queued above may already be running/completing concurrently
        // with these two blocking MPI_Sendrecv calls -- this is the
        // overlap the section is about.
        CUDA_CHECK(cudaStreamSynchronize(bottomStream));
        MPI_Sendrecv(d_output + (nyLocal - 2) * nx, nx, MPI_FLOAT, bottomNeighbor, 1,
                     d_output, nx, MPI_FLOAT, topNeighbor, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Fig. 23.14 line 37: host must wait for the cudaMemcpyAsync
        // above to complete before using l2norm_h in MPI_Allreduce.
        CUDA_CHECK(cudaStreamSynchronize(internalStream));

        float l2normSumSq_h = 0.0f;
        MPI_Allreduce(l2norm_h, &l2normSumSq_h, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        l2norm = std::sqrt(l2normSumSq_h);

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

    CUDA_CHECK(cudaEventDestroy(resetL2));
    CUDA_CHECK(cudaEventDestroy(topDone));
    CUDA_CHECK(cudaEventDestroy(bottomDone));
    CUDA_CHECK(cudaStreamDestroy(topStream));
    CUDA_CHECK(cudaStreamDestroy(bottomStream));
    CUDA_CHECK(cudaStreamDestroy(internalStream));
    CUDA_CHECK(cudaFreeHost(l2norm_h));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_l2normSq));

    MPI_Finalize();
    return 0;
}
