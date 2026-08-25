// Chapter 23: Multi-GPU programming
// Section 23.5 "Multi-GPU stencil with NVSHMEM" (Fig. 23.22-23.25).
//
// NOT COMPILED/RUN ON THIS MACHINE: no system-wide NVSHMEM development
// package is installed here (no <nvshmem.h>/<nvshmemx.h>). Same domain
// decomposition as files 02-04 (1D along y, periodic/wrap-around rank
// topology, x=0/x=nx-1 fixed Dirichlet edges), same interior-point Jacobi
// math -- but here the halo exchange is moved *inside* the kernel itself,
// using NVSHMEM one-sided put operations, per §23.5:
//
//   "An alternative form of communication that is commonly used in
//    parallel computing is one-sided communication. In one-sided
//    communication, a process can send data to or receive data from
//    another process without involving the other process... NVSHMEM
//    allows put and get operations to be initiated by the host threads or
//    by individual device threads of a kernel."  (§23.5)
//
// In NVSHMEM terminology a rank is called a "processing element" (PE).
// Unlike files 02-04, which launch three kernels (top/bottom boundary +
// internal) on three streams to overlap computation with communication,
// NVSHMEM needs only a *single* kernel launch on a *single* stream: the
// put calls that send boundary rows to neighboring PEs are issued directly
// by the threads that compute those boundary values, so computation and
// communication are fused into one kernel and overlap automatically
// (§23.5: "Using NVSHMEM to perform put operations of the halo values
// directly from the kernel has two key advantages. The first advantage is
// that it automatically overlaps computation and communication... it is
// unnecessary to launch separate kernels and orchestrate the concurrency
// between them.").

#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#include "../../common/cuda_utils.h"

// ---------------------------------------------------------------------
// §23.5, Fig. 23.23: Jacobi kernel with in-kernel NVSHMEM halo exchange.
// Differences from the plain jacobiKernel in files 01-04:
//   - takes the top/bottom PE ids as extra parameters (Fig. 23.23 line 03)
//   - after computing a thread's new value, if that thread's row is the
//     local top boundary row (y == 1), it uses nvshmem_float_p to place
//     the value directly into the *top PE's bottom halo row* (Fig. 23.23
//     lines 15-16); if it is the local bottom boundary row (y == ny-2), it
//     puts into the *bottom PE's top halo row* (lines 18-20).
// Both puts target the address of the corresponding halo element in the
// *sender's own* array; NVSHMEM's symmetric heap resolves that to the
// matching offset in the target PE's array because every PE allocates its
// input/output arrays identically via nvshmem_malloc (§23.5: "all
// processes allocate corresponding memory objects within their address
// spaces at the same offset in their respective local instances of the
// symmetric heap... for an initiating process to access an object in a
// target process' address space, it can simply provide the address of the
// corresponding object in its own address space").
// ---------------------------------------------------------------------
__global__ void jacobiKernelNvshmem(float* out, const float* in, int nx, int ny,
                                     float* l2normSq, int topPe, int bottomPe) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    float mySq = 0.0f;
    if (ix > 0 && ix < nx - 1 && iy > 0 && iy < ny - 1) {
        int idx = iy * nx + ix;
        float newVal = 0.25f * (in[idx - 1] + in[idx + 1] + in[idx - nx] + in[idx + nx]);
        out[idx] = newVal;
        float residue = newVal - in[idx];
        mySq = residue * residue;

        // Fig. 23.23 line 15: thread computing the top boundary row (y==1)
        // sends its new value into the top PE's bottom halo row (row ny-1
        // of the top PE's array).
        if (iy == 1) {
            nvshmem_float_p(&out[(ny - 1) * nx + ix], newVal, topPe);
        }
        // Fig. 23.23 line 18: thread computing the bottom boundary row
        // (y==ny-2) sends its new value into the bottom PE's top halo row
        // (row 0 of the bottom PE's array).
        if (iy == ny - 2) {
            nvshmem_float_p(&out[0 * nx + ix], newVal, bottomPe);
        }
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

static void launchJacobiKernelNvshmem(float* out, const float* in, int nx, int ny,
                                       float* l2normSq, int topPe, int bottomPe,
                                       cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    size_t shmem = static_cast<size_t>(block.x) * block.y * sizeof(float);
    jacobiKernelNvshmem<<<grid, block, shmem, stream>>>(out, in, nx, ny, l2normSq, topPe, bottomPe);
}

__global__ void initGridKernel(float* grid, int nx, int nyLocal) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < nx && iy < nyLocal) {
        // The multi-PE domain is periodic in y (see 02's file header), so
        // there is no single "hot" y-edge as in file 01. x=0/x=nx-1 remain
        // true, non-decomposed Dirichlet edges in every file since the
        // book never decomposes along x, so we drive the problem from a
        // Dirichlet edge at x=0 (analogous to file 01's y=0 edge) instead;
        // everything else is initialized to 0.
        grid[iy * nx + ix] = (ix == 0) ? 1.0f : 0.0f;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int mpiRank = 0, mpiSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    CUDA_CHECK(cudaSetDevice(mpiRank % deviceCount));

    // §23.5, Fig. 23.24: initialize NVSHMEM using an existing MPI
    // communicator. `mpi_comm` is a pointer field in nvshmemx_init_attr_t
    // ("mpi_comm, that points to an MPI communicator object"), so we must
    // give it the address of an MPI_Comm variable, not the value itself.
    MPI_Comm mpiComm = MPI_COMM_WORLD;
    nvshmemx_init_attr_t attr;
    attr.mpi_comm = &mpiComm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    int myPe = nvshmem_my_pe();
    int nPes = nvshmem_n_pes();

    const int nx = 130;
    const int nyTotalInterior = 512;
    const int nyLocalInterior = nyTotalInterior / nPes;
    const int nyLocal = nyLocalInterior + 2;

    const int maxIters = 2000;
    const float tol = 1e-6f;

    size_t bytes = static_cast<size_t>(nx) * nyLocal * sizeof(float);

    // §23.5: "input = (float*) nvshmem_malloc(nx*ny*sizeof(float));" --
    // symmetric heap allocation. All PEs must call nvshmem_malloc the same
    // number of times with the same sizes ("All PEs must participate when
    // calling these routines").
    float* input = static_cast<float*>(nvshmem_malloc(bytes));
    float* output = static_cast<float*>(nvshmem_malloc(bytes));
    if (!input || !output) {
        fprintf(stderr, "nvshmem_malloc failed on PE %d\n", myPe);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    float* d_l2normSq = nullptr;
    CUDA_CHECK(cudaMalloc(&d_l2normSq, sizeof(float)));
    float* l2norm_h = nullptr;
    CUDA_CHECK(cudaMallocHost(&l2norm_h, sizeof(float)));

    dim3 initBlock(16, 16);
    dim3 initGrid((nx + initBlock.x - 1) / initBlock.x, (nyLocal + initBlock.y - 1) / initBlock.y);
    initGridKernel<<<initGrid, initBlock>>>(input, nx, nyLocal);
    initGridKernel<<<initGrid, initBlock>>>(output, nx, nyLocal);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Same periodic wrap-around neighbor topology as files 02-04, but
    // expressed in PE ids per §23.5's terminology.
    int topPe = (myPe == 0) ? (nPes - 1) : (myPe - 1);
    int bottomPe = (myPe == nPes - 1) ? 0 : (myPe + 1);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    int iter = 0;
    float l2norm = 1e30f;
    while (iter < maxIters && l2norm > tol) {
        // Fig. 23.25 line 04: reset the L2 norm accumulator.
        CUDA_CHECK(cudaMemsetAsync(d_l2normSq, 0, sizeof(float), stream));

        // Fig. 23.25 lines 07-09: compute topPe/bottomPe (done once above,
        // since the wrap-around topology is fixed for the run) and launch
        // a *single* kernel for the whole local slab -- computation and
        // halo-exchange puts are fused in jacobiKernelNvshmem.
        launchJacobiKernelNvshmem(output, input, nx, nyLocal, d_l2normSq, topPe, bottomPe, stream);

        // §23.5, Fig. 23.25 line 18: nvshmemx_barrier_all_on_stream is
        // required because finishing the kernel only guarantees the put
        // operations were *initiated*, not that the data has *arrived* at
        // the target PE ("Synchronizing on the stream only guarantees that
        // the kernel has finished its computation and sending the data; it
        // does not guarantee that the data has been received... This call
        // makes all PEs wait at the barrier until all the NVSHMEM memory
        // access operations initiated by any preceding kernels in the
        // stream have completed.").
        nvshmemx_barrier_all_on_stream(stream);

        CUDA_CHECK(cudaMemcpyAsync(l2norm_h, d_l2normSq, sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Fig. 23.25 lines 12-16: same MPI_Allreduce-based L2 norm
        // reduction as the MPI/NCCL versions -- NVSHMEM replaces only the
        // halo exchange, not the L2-norm collective.
        float l2normSumSq_h = 0.0f;
        MPI_Allreduce(l2norm_h, &l2normSumSq_h, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        l2norm = std::sqrt(l2normSumSq_h);

        std::swap(input, output);
        ++iter;

        if (myPe == 0 && (iter % 500 == 0)) {
            printf("iter %d: l2norm = %.6e\n", iter, l2norm);
        }
    }

    if (myPe == 0) {
        printf("PE 0: finished after %d iterations, final L2 norm = %.6e\n", iter, l2norm);
        printf("(not executed on this machine -- see README)\n");
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(l2norm_h));
    CUDA_CHECK(cudaFree(d_l2normSq));
    nvshmem_free(input);
    nvshmem_free(output);
    nvshmem_finalize();

    MPI_Finalize();
    return 0;
}
