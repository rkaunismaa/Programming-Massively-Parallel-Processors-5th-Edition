// Chapter 21: Electrostatic potential map
// §21.4, Fig. 21.9-21.10: DCS coarsened gather kernel with coalesced writes.
//
// §21.4 profiles 03_dcs_coarsened.cu (Fig. 21.8) and finds its energygrid
// writes un-coalesced: each thread's COARSEN=4 output grid points are
// ADJACENT to each other (xBase, xBase+1, xBase+2, xBase+3), which means
// adjacent THREADS' outputs are 4 elements apart -- "the 32 locations to be
// written by all the threads in a warp are spread out, with three elements
// in between the loaded/written locations."
//
// The fix (Fig. 21.9): reassign which grid points go to which thread so
// that adjacent threads always write adjacent addresses. "We first assign
// blockDim.x consecutive grid points in the x dimension to the threads. We
// then assign the next blockDim.x consecutive grid points to the same
// threads. We repeat the assignment until each thread has the number of
// grid points desired." I.e. thread `threadIdx.x` within a block owns grid
// points at offsets `threadIdx.x + i*blockDim.x` for i = 0..COARSEN-1
// (relative to the block's base x), not `threadIdx.x*COARSEN + i`.
//
// Fig. 21.10's kernel is otherwise identical in its per-atom arithmetic and
// reuse (dy, dz, dysqdzsq, atom charge/x read once, reused across all
// COARSEN outputs) to Fig. 21.8 -- only the grid-point-to-thread mapping
// (and consequently the x-coordinate offsets, now `blockDim.x*gridspacing`
// apart instead of `gridspacing` apart, per §21.4: "the x-coordinates ...
// are offset by blockDim.x*gridspacing") and the resulting write pattern
// change. With this assignment, for any fixed i, address
// `base + threadIdx.x + i*blockDim.x` is contiguous across threadIdx.x, so
// all 4 writes per thread are now coalesced.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

#define GRID_X 96
#define GRID_Y 96
#define GRIDSPACING 0.5f
#define Z_SLICE 0.0f
#define COARSEN 4

#define NUM_ATOMS 4000
#define CHUNK_SIZE 1500

__constant__ float c_atoms[CHUNK_SIZE * 4];

// ---------------------------------------------------------------------------
// Fig. 21.10: one thread computes COARSEN=4 grid points spaced blockDim.x
// apart (interleaved across the block), so writes from a warp are coalesced.
// ---------------------------------------------------------------------------
__global__ void dcsCoalescedKernel(int numAtomsInChunk, float *energygrid) {
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gy >= GRID_Y) return;

    // Base x-index of this block's tile; the block covers blockDim.x*COARSEN
    // consecutive x-indices, tiled across gridDim.x blocks (Fig. 21.9).
    int xBlockBase = blockIdx.x * blockDim.x * COARSEN;
    int x0Idx = xBlockBase + threadIdx.x;
    if (x0Idx >= GRID_X) return;

    float y = gy * GRIDSPACING;
    float x0 = x0Idx * GRIDSPACING;
    float xStride = blockDim.x * GRIDSPACING;  // offset between a thread's own outputs

    float e[COARSEN];
#pragma unroll
    for (int i = 0; i < COARSEN; ++i) e[i] = 0.0f;

    for (int i = 0; i < numAtomsInChunk; ++i) {
        float ax = c_atoms[i * 4 + 0];
        float ay = c_atoms[i * 4 + 1];
        float az = c_atoms[i * 4 + 2];
        float charge = c_atoms[i * 4 + 3];

        float dy = y - ay;
        float dz = Z_SLICE - az;
        float dysqdzsq = dy * dy + dz * dz;

        float x = x0;
#pragma unroll
        for (int k = 0; k < COARSEN; ++k) {
            float dx = x - ax;
            e[k] += charge / sqrtf(dx * dx + dysqdzsq);
            x += xStride;
        }
    }

    int xIdx = x0Idx;
#pragma unroll
    for (int k = 0; k < COARSEN; ++k) {
        if (xIdx < GRID_X) energygrid[gy * GRID_X + xIdx] += e[k];
        xIdx += blockDim.x;
    }
}

static void cpuDcsReference(const std::vector<float> &atoms, std::vector<float> &grid) {
    grid.assign(static_cast<size_t>(GRID_X) * GRID_Y, 0.0f);
    for (int y = 0; y < GRID_Y; ++y) {
        float gy = y * GRIDSPACING;
        for (int x = 0; x < GRID_X; ++x) {
            float gx = x * GRIDSPACING;
            double energy = 0.0;
            for (int i = 0; i < NUM_ATOMS; ++i) {
                float ax = atoms[i * 4 + 0], ay = atoms[i * 4 + 1];
                float az = atoms[i * 4 + 2], charge = atoms[i * 4 + 3];
                float dx = gx - ax, dy = gy - ay, dz = Z_SLICE - az;
                energy += static_cast<double>(charge) / std::sqrt(static_cast<double>(dx * dx + dy * dy + dz * dz));
            }
            grid[y * GRID_X + x] = static_cast<float>(energy);
        }
    }
}

static std::vector<float> makeAtoms(unsigned int seed) {
    std::vector<float> atoms(static_cast<size_t>(NUM_ATOMS) * 4);
    unsigned int state = seed;
    auto nextRand = [&]() -> float {
        state = state * 1103515245u + 12345u;
        return static_cast<float>((state >> 8) & 0xFFFFFFu) / static_cast<float>(0xFFFFFFu);
    };
    float xMax = (GRID_X - 1) * GRIDSPACING;
    float yMax = (GRID_Y - 1) * GRIDSPACING;
    for (int i = 0; i < NUM_ATOMS; ++i) {
        atoms[i * 4 + 0] = nextRand() * xMax;
        atoms[i * 4 + 1] = nextRand() * yMax;
        atoms[i * 4 + 2] = 3.0f + nextRand() * 6.0f;
        atoms[i * 4 + 3] = -2.0f + nextRand() * 4.0f;
    }
    return atoms;
}

static bool checkClose(const std::vector<float> &a, const std::vector<float> &b, float eps, float *maxDiff) {
    bool ok = true;
    float md = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > md) md = diff;
        if (!nearlyEqual(a[i], b[i], eps)) ok = false;
    }
    if (maxDiff) *maxDiff = md;
    return ok;
}

int main() {
    static_assert(GRID_X % COARSEN == 0, "GRID_X must be a multiple of COARSEN");
    printf("DCS coarsened + coalesced gather kernel (§21.4, Fig. 21.9-21.10):\n");
    printf("grid=%dx%d spacing=%.2f atoms=%d chunk=%d coarsen=%d\n", GRID_X, GRID_Y, GRIDSPACING, NUM_ATOMS,
           CHUNK_SIZE, COARSEN);

    std::vector<float> atoms = makeAtoms(1u);
    std::vector<float> gridRef;
    cpuDcsReference(atoms, gridRef);

    float *grid_d;
    size_t gridBytes = static_cast<size_t>(GRID_X) * GRID_Y * sizeof(float);
    CUDA_CHECK(cudaMalloc(&grid_d, gridBytes));

    int numChunks = (NUM_ATOMS + CHUNK_SIZE - 1) / CHUNK_SIZE;
    dim3 block(8, 16);
    int numXBlocks = GRID_X / (static_cast<int>(block.x) * COARSEN);
    dim3 launchGrid(numXBlocks, (GRID_Y + block.y - 1) / block.y);

    auto launch = [&]() {
        CUDA_CHECK(cudaMemset(grid_d, 0, gridBytes));
        for (int c = 0; c < numChunks; ++c) {
            int chunkStart = c * CHUNK_SIZE;
            int chunkAtoms = std::min(CHUNK_SIZE, NUM_ATOMS - chunkStart);
            CUDA_CHECK(cudaMemcpyToSymbol(c_atoms, atoms.data() + static_cast<size_t>(chunkStart) * 4,
                                           static_cast<size_t>(chunkAtoms) * 4 * sizeof(float)));
            dcsCoalescedKernel<<<launchGrid, block>>>(chunkAtoms, grid_d);
            CUDA_CHECK(cudaGetLastError());
        }
    };

    launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    launch();
    float ms = timer.stopAndGetMs();

    std::vector<float> grid(static_cast<size_t>(GRID_X) * GRID_Y);
    CUDA_CHECK(cudaMemcpy(grid.data(), grid_d, gridBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(grid_d));

    float maxDiff = 0.0f;
    bool ok = checkClose(grid, gridRef, 1e-3f, &maxDiff);
    printf("GPU vs CPU reference: max|diff|=%.6f  %.4f ms  [%s]\n", maxDiff, ms, ok ? "match" : "MISMATCH");

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
