// Chapter 23: Multi-GPU programming
// Section 23.1 "Stencil as a running example" (Fig. 23.1).
//
// This is the chapter's single-GPU running example: the 2D Jacobi iterative
// method stencil that every later section (23.2 MPI, 23.3 MPI+overlap,
// 23.4 NCCL, 23.5 NVSHMEM) distributes across multiple GPUs. At this point
// in the chapter there is no domain decomposition yet -- one GPU owns the
// entire nx*ny grid, so there are no halo rows to exchange with a neighbor;
// the array's true edges (x=0, x=nx-1, y=0, y=ny-1) are genuine Dirichlet
// boundary points, matching the book's description exactly:
//
//   "each thread identifies the x and y coordinates of the grid point it is
//    responsible for computing... A boundary check makes sure that only
//    threads computing grid points that are within bounds are active...
//    the points at x=0, x=nx-1, y=0, and y=ny-1 are skipped because they
//    are either boundary points or halo points, as we will see later...
//    The thread calculates the new value at its grid point as the average
//    of the old values at the neighboring grid points... calculates its
//    grid point's residue as the difference between the new value and the
//    old value... A block-wide reduction is performed to sum up the
//    squares of the residues across the entire block... one thread in the
//    block atomically accumulates the block-level sum to the grid-wide L2
//    norm."  (§23.1)
//
// The kernel below is written with a generalized `numRows` span parameter
// (rather than hard-coding `ny`) because the *exact same* kernel is reused
// unmodified in 02/03/04 (multi-GPU MPI/overlap/NCCL) to compute a rank's
// full local slab (numRows = local ny) or a sub-span of it (numRows = 3 for
// a single boundary row, or numRows = ny-2 for the internal rows) -- see
// those files' comments for how §23.3's Fig. 23.14 splits this same update
// into three kernel launches. This file always calls it with the full local
// grid (numRows = ny), which is exactly Fig. 23.1's single-GPU kernel.
//
// Test problem (not specified numerically by the book, which only gives the
// generic "modeled system" stencil): a 2D Laplace/steady-state-heat problem
// with Dirichlet boundary conditions -- the top edge (y=0) held at 1.0, the
// other three edges held at 0.0, interior initialized to 0.0. This is a
// standard, non-arbitrary test for a Jacobi relaxation stencil (converges
// monotonically to a fixed point, so it is a strong, deterministic
// correctness check between two independent implementations run for the
// same number of iterations).
//
// Correctness is checked against an independently written CPU reference
// (double precision, same iteration structure) run for the identical
// number of iterations, compared point-by-point with nearlyEqual().

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#include "../../common/cuda_utils.h"

// ---------------------------------------------------------------------
// GPU kernel: one Jacobi update + residual reduction over a `numRows`-row
// span of the grid (rows 0 and numRows-1 of the given span are treated as
// fixed boundary/halo input rows and are not written).
// ---------------------------------------------------------------------
__global__ void jacobiKernel(float* out, const float* in, int nx, int numRows,
                              float* l2normSq) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    float mySq = 0.0f;
    if (ix > 0 && ix < nx - 1 && iy > 0 && iy < numRows - 1) {
        int idx = iy * nx + ix;
        // Average of the 4 neighboring grid points (5-point stencil),
        // per Fig. 23.1 lines 08-12.
        float newVal = 0.25f * (in[idx - 1] + in[idx + 1] + in[idx - nx] + in[idx + nx]);
        out[idx] = newVal;
        float residue = newVal - in[idx];  // Fig. 23.1 line 13
        mySq = residue * residue;
    }

    // Block-wide reduction of the squared residues (Fig. 23.1 line 16),
    // then one atomicAdd per block into the grid-wide L2 norm accumulator
    // (Fig. 23.1 lines 17-19).
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

__global__ void initGridKernel(float* grid, int nx, int ny) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix < nx && iy < ny) {
        grid[iy * nx + ix] = (iy == 0) ? 1.0f : 0.0f;
    }
}

// ---------------------------------------------------------------------
// Independent CPU reference: same Jacobi iteration, same boundary
// conditions, double precision, run for the identical number of
// iterations as the GPU loop below.
// ---------------------------------------------------------------------
static void cpuJacobiReference(std::vector<double>& result, int nx, int ny,
                                int maxIters, double tol, int* itersUsed,
                                double* finalNorm) {
    std::vector<double> a(static_cast<size_t>(nx) * ny);
    std::vector<double> b(static_cast<size_t>(nx) * ny);
    for (int iy = 0; iy < ny; ++iy) {
        for (int ix = 0; ix < nx; ++ix) {
            a[iy * nx + ix] = (iy == 0) ? 1.0 : 0.0;
        }
    }
    b = a;

    double* cur = a.data();
    double* nxt = b.data();
    int iter = 0;
    double norm = 1e300;
    while (iter < maxIters && norm > tol) {
        double sumSq = 0.0;
        for (int iy = 1; iy < ny - 1; ++iy) {
            for (int ix = 1; ix < nx - 1; ++ix) {
                int idx = iy * nx + ix;
                double newVal = 0.25 * (cur[idx - 1] + cur[idx + 1] + cur[idx - nx] + cur[idx + nx]);
                nxt[idx] = newVal;
                double residue = newVal - cur[idx];
                sumSq += residue * residue;
            }
        }
        norm = std::sqrt(sumSq);
        std::swap(cur, nxt);
        ++iter;
    }

    result.assign(cur, cur + static_cast<size_t>(nx) * ny);
    *itersUsed = iter;
    *finalNorm = norm;
}

int main() {
    const int nx = 130;
    const int ny = 130;
    const int maxIters = 2000;
    const float tol = 1e-6f;

    size_t bytes = static_cast<size_t>(nx) * ny * sizeof(float);

    float *d_input = nullptr, *d_output = nullptr, *d_l2normSq = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMalloc(&d_l2normSq, sizeof(float)));

    dim3 initBlock(16, 16);
    dim3 initGrid((nx + initBlock.x - 1) / initBlock.x, (ny + initBlock.y - 1) / initBlock.y);
    initGridKernel<<<initGrid, initBlock>>>(d_input, nx, ny);
    initGridKernel<<<initGrid, initBlock>>>(d_output, nx, ny);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();

    int iter = 0;
    float l2norm = 1e30f;
    while (iter < maxIters && l2norm > tol) {
        CUDA_CHECK(cudaMemset(d_l2normSq, 0, sizeof(float)));
        launchJacobiKernel(d_output, d_input, nx, ny, d_l2normSq);
        CUDA_CHECK(cudaGetLastError());

        float l2normSq_h = 0.0f;
        CUDA_CHECK(cudaMemcpy(&l2normSq_h, d_l2normSq, sizeof(float), cudaMemcpyDeviceToHost));
        l2norm = std::sqrt(l2normSq_h);

        std::swap(d_input, d_output);
        ++iter;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    float gpuMs = timer.stopAndGetMs();

    // After the last swap, the most recently computed grid is in d_input.
    std::vector<float> gpuResult(static_cast<size_t>(nx) * ny);
    CUDA_CHECK(cudaMemcpy(gpuResult.data(), d_input, bytes, cudaMemcpyDeviceToHost));

    std::vector<double> cpuResult;
    int cpuIters = 0;
    double cpuNorm = 0.0;
    cpuJacobiReference(cpuResult, nx, ny, maxIters, static_cast<double>(tol), &cpuIters, &cpuNorm);

    bool pass = true;
    double maxDiff = 0.0;
    for (size_t i = 0; i < gpuResult.size(); ++i) {
        double diff = std::fabs(static_cast<double>(gpuResult[i]) - cpuResult[i]);
        maxDiff = std::max(maxDiff, diff);
        if (!nearlyEqual(gpuResult[i], static_cast<float>(cpuResult[i]))) {
            pass = false;
        }
    }

    printf("Grid: %dx%d (interior %dx%d)\n", nx, ny, nx - 2, ny - 2);
    printf("GPU: %d iterations, final L2 norm = %.6e\n", iter, l2norm);
    printf("CPU: %d iterations, final L2 norm = %.6e\n", cpuIters, cpuNorm);
    printf("Max |GPU - CPU| = %.6e\n", maxDiff);
    printf("GPU time: %.3f ms (%d iterations)\n", gpuMs, iter);
    printf("%s\n", pass ? "PASS" : "FAIL");

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_l2normSq));

    return pass ? 0 : 1;
}
