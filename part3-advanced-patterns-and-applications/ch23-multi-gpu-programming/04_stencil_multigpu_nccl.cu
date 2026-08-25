// Chapter 23: Multi-GPU programming
// Section 23.4 "Multi-GPU stencil with NCCL" (Fig. 23.16-23.21).
//
// NOT COMPILED/RUN ON THIS MACHINE: no system-wide NCCL development
// package is installed here (no <nccl.h>). This file builds directly on
// 03_stencil_multigpu_mpi_overlap.cu: identical domain decomposition,
// identical three-kernel (top/bottom boundary + internal) split across
// topStream/bottomStream/internalStream, identical stream-priority setup.
// The only change is *how the halo exchange is expressed and synchronized*
// -- MPI_Sendrecv (host-blocking) is replaced by ncclSend/ncclRecv (stream-
// ordered, non-blocking from the host's point of view), per §23.4:
//
//   "an important difference is that NCCL communication primitives run on
//    the GPU, not the host CPU, and can be placed inside of CUDA streams.
//    Hence, using NCCL frees the host thread from having to serve as a
//    synchronizing intermediary between kernel calls and communication
//    primitives."  (§23.4)
//
// NCCL is used only for the halo exchange; MPI is still used for process
// bootstrap and for MPI_Bcast/MPI_Barrier/MPI_Allreduce, per §23.4: "NCCL
// is not a complete replacement for MPI, but rather, a library that can be
// used in conjunction with MPI to enhance multi-GPU communication."

#include <mpi.h>
#include <nccl.h>

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
        // The multi-rank domain is periodic in y (see 02's file header),
        // so there is no single "hot" y-edge as in file 01. x=0/x=nx-1
        // remain true, non-decomposed Dirichlet edges in every file since
        // the book never decomposes along x, so we drive the problem from
        // a Dirichlet edge at x=0 (analogous to file 01's y=0 edge)
        // instead; everything else is initialized to 0.
        grid[iy * nx + ix] = (ix == 0) ? 1.0f : 0.0f;
    }
}

#define NCCL_CHECK(call)                                                       \
    do {                                                                       \
        ncclResult_t res__ = (call);                                           \
        if (res__ != ncclSuccess) {                                            \
            fprintf(stderr, "NCCL error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    ncclGetErrorString(res__));                                \
            exit(EXIT_FAILURE);                                                \
        }                                                                       \
    } while (0)

// Modeled directly on NCCL_CHECK above: checks an MPI call's int return
// code against MPI_SUCCESS, prints an error, and aborts the whole MPI job
// on failure (MPI_Abort rather than exit(), since a bare exit() on one
// rank can hang the other ranks waiting in a collective call).
#define MPI_CHECK(call)                                                       \
    do {                                                                       \
        int err__ = (call);                                                    \
        if (err__ != MPI_SUCCESS) {                                            \
            fprintf(stderr, "MPI error at %s:%d: code %d\n", __FILE__, __LINE__, \
                    err__);                                                    \
            MPI_Abort(MPI_COMM_WORLD, err__);                                  \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------
// Independent CPU reference: ported from 01_stencil_singlegpu_baseline.cu's
// cpuJacobiReference (same as 02/03's cpuJacobiReferenceGlobal), adapted
// for this file's global topology -- the same Jacobi relaxation run over
// the *whole* nx x nyTotal grid (nyTotal = nyTotalInterior, the sum of
// every rank's local interior rows), with the y-dimension treated as
// periodic (a torus, matching the wrap-around rank topology) and
// x=0/x=nx-1 held at the same fixed Dirichlet values as initGridKernel
// above. Double precision, run for the identical number of iterations as
// the GPU loop below.
// ---------------------------------------------------------------------
static void cpuJacobiReferenceGlobal(std::vector<double>& result, int nx, int nyTotal,
                                      int maxIters, double tol, int* itersUsed,
                                      double* finalNorm) {
    std::vector<double> a(static_cast<size_t>(nx) * nyTotal);
    std::vector<double> b(static_cast<size_t>(nx) * nyTotal);
    for (int iy = 0; iy < nyTotal; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            a[iy * nx + ix] = (ix == 0) ? 1.0 : 0.0;
        }
    }
    b = a;

    double* cur = a.data();
    double* nxt = b.data();
    int iter = 0;
    double norm = 1e300;
    while (iter < maxIters && norm > tol) {
        double sumSq = 0.0;
        for (int iy = 0; iy < nyTotal; ++iy) {
            int iyUp = (iy - 1 + nyTotal) % nyTotal;
            int iyDown = (iy + 1) % nyTotal;
            for (int ix = 1; ix < nx - 1; ++ix) {
                int idx = iy * nx + ix;
                double newVal = 0.25 * (cur[idx - 1] + cur[idx + 1] +
                                         cur[iyUp * nx + ix] + cur[iyDown * nx + ix]);
                nxt[idx] = newVal;
                double residue = newVal - cur[idx];
                sumSq += residue * residue;
            }
        }
        norm = std::sqrt(sumSq);
        std::swap(cur, nxt);
        ++iter;
    }

    result.assign(cur, cur + static_cast<size_t>(nx) * nyTotal);
    *itersUsed = iter;
    *finalNorm = norm;
}

int main(int argc, char** argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));

    int rank = 0, numRanks = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numRanks));

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices visible (cudaGetDeviceCount "
                         "returned 0); check CUDA_VISIBLE_DEVICES\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));

    // §23.4, Fig. 23.16-23.17: bootstrap a NCCL communicator on top of the
    // MPI ranks -- rank 0 creates a unique ID, broadcasts it to everyone
    // via MPI_Bcast, all ranks barrier, then all ranks call
    // ncclCommInitRank with the same ID and their MPI rank as their NCCL
    // rank (valid because there is a 1-to-1 mapping between MPI ranks and
    // CUDA devices here, as the book assumes: "If there is a 1-to-1
    // mapping between MPI processes/threads and CUDA devices in an
    // application, one can simply use their MPI ranks for the NCCL
    // ranks").
    ncclUniqueId id;
    if (rank == 0) {
        NCCL_CHECK(ncclGetUniqueId(&id));
    }
    MPI_CHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, numRanks, id, rank));

    const int nx = 130;
    const int nyTotalInterior = 512;  // must be divisible by numRanks
    if (nyTotalInterior % numRanks != 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: nyTotalInterior (%d) must be divisible by "
                             "numRanks (%d)\n", nyTotalInterior, numRanks);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int nyLocalInterior = nyTotalInterior / numRanks;
    const int nyLocal = nyLocalInterior + 2;

    const int maxIters = 2000;
    const float tol = 1e-6f;

    size_t bytes = static_cast<size_t>(nx) * nyLocal * sizeof(float);

    float *d_input = nullptr, *d_output = nullptr, *d_l2normSq = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMalloc(&d_l2normSq, sizeof(float)));

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

    int lowPriority = 0, highPriority = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lowPriority, &highPriority));
    cudaStream_t topStream, bottomStream, internalStream;
    CUDA_CHECK(cudaStreamCreateWithPriority(&internalStream, cudaStreamDefault, lowPriority));
    CUDA_CHECK(cudaStreamCreateWithPriority(&topStream, cudaStreamDefault, highPriority));
    CUDA_CHECK(cudaStreamCreateWithPriority(&bottomStream, cudaStreamDefault, highPriority));

    // resetL2: same role as file 03. exchangeTop/exchangeBottom: recorded
    // in topStream/bottomStream *after* their ncclSend/ncclRecv group
    // calls, so waiting on them (from internalStream) guarantees both
    // "the boundary kernel finished" and "the halo data has actually been
    // sent/received" -- unlike file 03, no separate topDone/bottomDone
    // event is needed because the NCCL calls are themselves stream-ordered
    // after the boundary kernels in the same stream (§23.4: "we pass the
    // streams to ncclSend and ncclRecv as their last parameter. The CUDA
    // runtime ensures that these communication primitives are initiated
    // after the preceding kernels in the streams finish computing the
    // corresponding halo rows").
    cudaEvent_t resetL2, exchangeTop, exchangeBottom;
    CUDA_CHECK(cudaEventCreateWithFlags(&resetL2, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&exchangeTop, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&exchangeBottom, cudaEventDisableTiming));

    GpuTimer timer;
    timer.start();

    int iter = 0;
    float l2norm = 1e30f;
    while (iter < maxIters && l2norm > tol) {
        CUDA_CHECK(cudaMemsetAsync(d_l2normSq, 0, sizeof(float), internalStream));
        CUDA_CHECK(cudaEventRecord(resetL2, internalStream));
        CUDA_CHECK(cudaStreamWaitEvent(topStream, resetL2, 0));
        CUDA_CHECK(cudaStreamWaitEvent(bottomStream, resetL2, 0));

        // Same three-way kernel split as file 03 (Fig. 23.14 lines 10,
        // 12-13, 15-16), unchanged by the switch to NCCL.
        launchJacobiKernel(d_output, d_input, nx, 3, d_l2normSq, topStream);
        launchJacobiKernel(d_output + (nyLocal - 3) * nx, d_input + (nyLocal - 3) * nx,
                            nx, 3, d_l2normSq, bottomStream);
        launchJacobiKernel(d_output + nx, d_input + nx, nx, nyLocal - 2, d_l2normSq, internalStream);

        // §23.4, Fig. 23.18-23.19, Fig. 23.20 lines 25-36: halo exchange
        // with ncclSend/ncclRecv sandwiched between ncclGroupStart/
        // ncclGroupEnd (NCCL has no fused send+receive primitive like
        // MPI_Sendrecv, so a group call is used to fuse them and avoid
        // deadlock, per §23.4: "a fused send and receive operation can be
        // achieved by placing calls to ncclSend and ncclRecv in between
        // calls to ncclGroupStart and ncclGroupEnd").
        //
        // Group 1, on topStream (replaces file 03's first MPI_Sendrecv,
        // which depended on the top-boundary kernel in topStream): send my
        // new top boundary row to my top neighbor, receive my bottom halo
        // row from my bottom neighbor.
        NCCL_CHECK(ncclGroupStart());
        NCCL_CHECK(ncclSend(d_output + 1 * nx, nx, ncclFloat, topNeighbor, comm, topStream));
        NCCL_CHECK(ncclRecv(d_output + (nyLocal - 1) * nx, nx, ncclFloat, bottomNeighbor, comm, topStream));
        NCCL_CHECK(ncclGroupEnd());
        CUDA_CHECK(cudaEventRecord(exchangeTop, topStream));

        // Group 2, on bottomStream (replaces file 03's second
        // MPI_Sendrecv, which depended on the bottom-boundary kernel in
        // bottomStream): send my new bottom boundary row to my bottom
        // neighbor, receive my top halo row from my top neighbor.
        NCCL_CHECK(ncclGroupStart());
        NCCL_CHECK(ncclSend(d_output + (nyLocal - 2) * nx, nx, ncclFloat, bottomNeighbor, comm, bottomStream));
        NCCL_CHECK(ncclRecv(d_output, nx, ncclFloat, topNeighbor, comm, bottomStream));
        NCCL_CHECK(ncclGroupEnd());
        CUDA_CHECK(cudaEventRecord(exchangeBottom, bottomStream));

        // §23.4, Fig. 23.20 lines 43-44: internalStream waits on both
        // exchange-complete events before the host is allowed to use the
        // L2 norm -- no host-side cudaStreamSynchronize on topStream or
        // bottomStream is needed at all, unlike file 03 (this is the
        // "host is now free from the communication and synchronization
        // operations" benefit of §23.4).
        CUDA_CHECK(cudaStreamWaitEvent(internalStream, exchangeTop, 0));
        CUDA_CHECK(cudaStreamWaitEvent(internalStream, exchangeBottom, 0));

        CUDA_CHECK(cudaMemcpyAsync(l2norm_h, d_l2normSq, sizeof(float), cudaMemcpyDeviceToHost, internalStream));

        // The one remaining host-side synchronization point (§23.4,
        // closing paragraph): wait for internalStream before MPI_Allreduce.
        CUDA_CHECK(cudaStreamSynchronize(internalStream));

        float l2normSumSq_h = 0.0f;
        MPI_CHECK(MPI_Allreduce(l2norm_h, &l2normSumSq_h, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD));
        l2norm = std::sqrt(l2normSumSq_h);

        std::swap(d_input, d_output);
        ++iter;

        if (rank == 0 && (iter % 500 == 0)) {
            printf("iter %d: l2norm = %.6e\n", iter, l2norm);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    float gpuMs = timer.stopAndGetMs();

    // Gather each rank's final local interior slab (rows 1..nyLocal-2,
    // excluding the two halo rows) from device to host, then MPI_Gather
    // them all into one global nx x nyTotalInterior buffer on rank 0. This
    // is a collective call, so every rank must issue it (not just rank 0),
    // even though only rank 0 uses the assembled result.
    std::vector<float> localInterior(static_cast<size_t>(nx) * nyLocalInterior);
    CUDA_CHECK(cudaMemcpy(localInterior.data(), d_input + nx,
                           localInterior.size() * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<float> globalResult;
    if (rank == 0) {
        globalResult.resize(static_cast<size_t>(nx) * nyTotalInterior);
    }
    MPI_CHECK(MPI_Gather(localInterior.data(), nx * nyLocalInterior, MPI_FLOAT,
                         rank == 0 ? globalResult.data() : nullptr,
                         nx * nyLocalInterior, MPI_FLOAT, 0, MPI_COMM_WORLD));

    int exitCode = 0;
    if (rank == 0) {
        std::vector<double> cpuResult;
        int cpuIters = 0;
        double cpuNorm = 0.0;
        cpuJacobiReferenceGlobal(cpuResult, nx, nyTotalInterior, maxIters,
                                  static_cast<double>(tol), &cpuIters, &cpuNorm);

        bool pass = true;
        double maxDiff = 0.0;
        for (size_t i = 0; i < globalResult.size(); ++i) {
            double diff = std::fabs(static_cast<double>(globalResult[i]) - cpuResult[i]);
            maxDiff = std::max(maxDiff, diff);
            if (!nearlyEqual(globalResult[i], static_cast<float>(cpuResult[i]))) {
                pass = false;
            }
        }

        printf("Global grid: %dx%d across %d rank(s) (interior %dx%d)\n", nx,
               nyTotalInterior, numRanks, nx - 2, nyTotalInterior);
        printf("GPU: %d iterations, final L2 norm = %.6e\n", iter, l2norm);
        printf("CPU: %d iterations, final L2 norm = %.6e\n", cpuIters, cpuNorm);
        printf("Max |GPU - CPU| = %.6e\n", maxDiff);
        printf("GPU time: %.3f ms (%d iterations)\n", gpuMs, iter);
        printf("%s\n", pass ? "PASS" : "FAIL");
        // Not executed on this machine (no system-wide NCCL dev package
        // installed here -- see the file header and README): the verdict
        // above is what this comparison would print if run for real.
        exitCode = pass ? 0 : 1;
    }

    CUDA_CHECK(cudaEventDestroy(resetL2));
    CUDA_CHECK(cudaEventDestroy(exchangeTop));
    CUDA_CHECK(cudaEventDestroy(exchangeBottom));
    CUDA_CHECK(cudaStreamDestroy(topStream));
    CUDA_CHECK(cudaStreamDestroy(bottomStream));
    CUDA_CHECK(cudaStreamDestroy(internalStream));
    CUDA_CHECK(cudaFreeHost(l2norm_h));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_l2normSq));

    NCCL_CHECK(ncclCommDestroy(comm));
    MPI_CHECK(MPI_Finalize());
    return exitCode;
}
