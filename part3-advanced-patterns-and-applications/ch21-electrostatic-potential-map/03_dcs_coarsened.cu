// Chapter 21: Electrostatic potential map
// §21.3, Fig. 21.7-21.8: DCS gather kernel with thread coarsening.
//
// §21.3 starts from the gather kernel of 02_dcs_gather.cu (Fig. 21.6): each
// thread reads 4 constant-memory floats (atom x,y,z,charge) per atom to do
// 9 floating-point ops for one grid point. Fig. 21.7 observes that grid
// points sharing a row (same y) also share the y-component of the
// atom-to-gridpoint distance, so folding COARSEN_FACTOR=4 grid points from
// the same row into one thread lets that thread fetch each atom's x, y, z,
// charge from constant memory ONCE and reuse them across all 4 outputs,
// per Fig. 21.8's exact reuse pattern:
//   - dy = y - atom.y                     computed once per atom
//   - dz = Z_SLICE - atom.z               computed once per atom
//   - dysqdzsq = dy*dy + dz*dz            computed once per atom, reused
//     by all 4 energy accumulators
//   - atom.charge                         read once, reused by all 4
//   - dx0..dx3 = x0..x3 - atom.x          atom.x read once, reused for all
//     4 dx_i (Fig. 21.8's stated savings: "eliminates three accesses to
//     constant memory for the y coordinate ... three for the x
//     coordinate ... three for the charge ... when processing an atom for
//     four grid points" -- i.e. 4 constant accesses total per atom here,
//     vs. 16 for four independent Fig. 21.6 threads).
//
// This file deliberately reproduces Fig. 21.8's grid-point assignment
// as-is: the four grid points folded into one thread are the four
// ADJACENT x-indices (xBase, xBase+1, xBase+2, xBase+3). §21.4 (see
// 04_dcs_coalesced.cu) is what identifies and fixes the resulting
// un-coalesced write pattern -- this file intentionally keeps the
// pre-§21.4 layout so the two files show the "before" and "after" of that
// optimization, matching the chapter's own narrative order.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

#define GRID_X 96
#define GRID_Y 96
#define GRIDSPACING 0.5f
#define Z_SLICE 0.0f
#define COARSEN 4  // Fig. 21.8 coarsens 4 grid points per thread

#define NUM_ATOMS 4000
#define CHUNK_SIZE 1500

__constant__ float c_atoms[CHUNK_SIZE * 4];

// ---------------------------------------------------------------------------
// Fig. 21.8: one thread computes COARSEN=4 ADJACENT grid points along x
// (xBase..xBase+3), reusing each atom's y/z/charge/x reads across all 4.
// ---------------------------------------------------------------------------
__global__ void dcsCoarsenedKernel(int numAtomsInChunk, float *energygrid) {
    int xGroup = blockIdx.x * blockDim.x + threadIdx.x;  // one group = COARSEN grid points
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    int numXGroups = GRID_X / COARSEN;
    if (xGroup >= numXGroups || gy >= GRID_Y) return;

    int xBase = xGroup * COARSEN;
    float y = gy * GRIDSPACING;
    float x0 = xBase * GRIDSPACING;
    float x1 = x0 + GRIDSPACING;
    float x2 = x0 + 2.0f * GRIDSPACING;
    float x3 = x0 + 3.0f * GRIDSPACING;

    float e0 = 0.0f, e1 = 0.0f, e2 = 0.0f, e3 = 0.0f;
    for (int i = 0; i < numAtomsInChunk; ++i) {
        float ax = c_atoms[i * 4 + 0];
        float ay = c_atoms[i * 4 + 1];
        float az = c_atoms[i * 4 + 2];
        float charge = c_atoms[i * 4 + 3];

        float dy = y - ay;
        float dz = Z_SLICE - az;
        float dysqdzsq = dy * dy + dz * dz;

        float dx0 = x0 - ax, dx1 = x1 - ax, dx2 = x2 - ax, dx3 = x3 - ax;
        e0 += charge / sqrtf(dx0 * dx0 + dysqdzsq);
        e1 += charge / sqrtf(dx1 * dx1 + dysqdzsq);
        e2 += charge / sqrtf(dx2 * dx2 + dysqdzsq);
        e3 += charge / sqrtf(dx3 * dx3 + dysqdzsq);
    }

    int base = gy * GRID_X + xBase;
    energygrid[base + 0] += e0;
    energygrid[base + 1] += e1;
    energygrid[base + 2] += e2;
    energygrid[base + 3] += e3;
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
    printf("DCS thread-coarsened gather kernel (§21.3, Fig. 21.7-21.8):\n");
    printf("grid=%dx%d spacing=%.2f atoms=%d chunk=%d coarsen=%d\n", GRID_X, GRID_Y, GRIDSPACING, NUM_ATOMS,
           CHUNK_SIZE, COARSEN);

    std::vector<float> atoms = makeAtoms(1u);
    std::vector<float> gridRef;
    cpuDcsReference(atoms, gridRef);

    float *grid_d;
    size_t gridBytes = static_cast<size_t>(GRID_X) * GRID_Y * sizeof(float);
    CUDA_CHECK(cudaMalloc(&grid_d, gridBytes));

    int numChunks = (NUM_ATOMS + CHUNK_SIZE - 1) / CHUNK_SIZE;
    int numXGroups = GRID_X / COARSEN;
    dim3 block(8, 16);
    dim3 launchGrid((numXGroups + block.x - 1) / block.x, (GRID_Y + block.y - 1) / block.y);

    auto launch = [&]() {
        CUDA_CHECK(cudaMemset(grid_d, 0, gridBytes));
        for (int c = 0; c < numChunks; ++c) {
            int chunkStart = c * CHUNK_SIZE;
            int chunkAtoms = std::min(CHUNK_SIZE, NUM_ATOMS - chunkStart);
            CUDA_CHECK(cudaMemcpyToSymbol(c_atoms, atoms.data() + static_cast<size_t>(chunkStart) * 4,
                                           static_cast<size_t>(chunkAtoms) * 4 * sizeof(float)));
            dcsCoarsenedKernel<<<launchGrid, block>>>(chunkAtoms, grid_d);
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
