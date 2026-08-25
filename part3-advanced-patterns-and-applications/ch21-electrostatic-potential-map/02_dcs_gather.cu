// Chapter 21: Electrostatic potential map
// §21.2, Fig. 21.3, Fig. 21.6: Direct Coulomb Summation (DCS), gather kernel.
//
// §21.2 contrasts the scatter kernel (01_dcs_scatter.cu, Fig. 21.5, one
// thread per atom, atomic writes) with a gather kernel: one thread per grid
// point, reading (gathering) all atoms' contributions with no atomics,
// since each thread owns exactly one output element. This requires going
// back to Fig. 21.3's unoptimized loop order (y outer, x next, atoms
// innermost) rather than Fig. 21.4's atom-outer order, because the thread
// grid needs to be indexed by grid point, not by atom -- the chapter's own
// framing of the tradeoff: "we would be parallelizing a slower C
// implementation".
//
// Fig. 21.6's kernel construction, applied to Fig. 21.3:
//   - the two outer loops (over y then x) are replaced by a 2D thread grid
//     that matches the 2D potential-grid shape one-to-one (§21.2: "We form
//     a two-dimensional thread grid that matches the two-dimensional
//     potential grid point organization");
//   - each thread computes its own gy, gx once (the chapter notes this is
//     the price of flattening the 2-level loop into one thread-grid
//     iteration -- y is now recomputed redundantly by every thread in a
//     row instead of once per row, "a tradeoff between the amount of
//     calculation done and the level of parallelism achieved");
//   - the atoms loop (Fig. 21.3 line 09) stays innermost inside the
//     kernel, iterating the current constant-memory chunk with no reuse
//     of dx/dy/dz across atoms (that reuse is what Fig. 21.4/Fig. 21.5
//     traded away to get the atom-outer scatter order);
//   - the thread accumulates into its own grid point with plain += (no
//     atomics needed) and the kernel is invoked once per atom chunk,
//     accumulating into the same device grid buffer across chunks --
//     exactly the same host-side constant-memory chunking scheme as the
//     scatter kernel (§21.2).

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

#define GRID_X 96
#define GRID_Y 96
#define GRIDSPACING 0.5f
#define Z_SLICE 0.0f

#define NUM_ATOMS 4000
#define CHUNK_SIZE 1500

__constant__ float c_atoms[CHUNK_SIZE * 4];

// ---------------------------------------------------------------------------
// Fig. 21.6: one thread per grid point; the atoms loop is innermost, kept
// exactly as in the unoptimized Fig. 21.3 (no cross-atom reuse). Called
// once per atom chunk, accumulating into energygrid across calls.
// ---------------------------------------------------------------------------
__global__ void dcsGatherKernel(int numAtomsInChunk, float *energygrid) {
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= GRID_X || gy >= GRID_Y) return;

    float x = gx * GRIDSPACING;
    float y = gy * GRIDSPACING;

    float energy = 0.0f;
    for (int i = 0; i < numAtomsInChunk; ++i) {
        float dx = x - c_atoms[i * 4 + 0];
        float dy = y - c_atoms[i * 4 + 1];
        float dz = Z_SLICE - c_atoms[i * 4 + 2];
        float charge = c_atoms[i * 4 + 3];
        energy += charge / sqrtf(dx * dx + dy * dy + dz * dz);
    }
    energygrid[gy * GRID_X + gx] += energy;
}

// ---------------------------------------------------------------------------
// CPU reference: same unoptimized triple loop as 01_dcs_scatter.cu.
// ---------------------------------------------------------------------------
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
    printf("DCS gather kernel (§21.2, Fig. 21.3, Fig. 21.6):\n");
    printf("grid=%dx%d spacing=%.2f atoms=%d chunk=%d\n", GRID_X, GRID_Y, GRIDSPACING, NUM_ATOMS, CHUNK_SIZE);

    std::vector<float> atoms = makeAtoms(1u);
    std::vector<float> gridRef;
    cpuDcsReference(atoms, gridRef);

    float *grid_d;
    size_t gridBytes = static_cast<size_t>(GRID_X) * GRID_Y * sizeof(float);
    CUDA_CHECK(cudaMalloc(&grid_d, gridBytes));

    int numChunks = (NUM_ATOMS + CHUNK_SIZE - 1) / CHUNK_SIZE;
    dim3 block(16, 16);
    dim3 launchGrid((GRID_X + block.x - 1) / block.x, (GRID_Y + block.y - 1) / block.y);

    auto launch = [&]() {
        CUDA_CHECK(cudaMemset(grid_d, 0, gridBytes));
        for (int c = 0; c < numChunks; ++c) {
            int chunkStart = c * CHUNK_SIZE;
            int chunkAtoms = std::min(CHUNK_SIZE, NUM_ATOMS - chunkStart);
            CUDA_CHECK(cudaMemcpyToSymbol(c_atoms, atoms.data() + static_cast<size_t>(chunkStart) * 4,
                                           static_cast<size_t>(chunkAtoms) * 4 * sizeof(float)));
            dcsGatherKernel<<<launchGrid, block>>>(chunkAtoms, grid_d);
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
