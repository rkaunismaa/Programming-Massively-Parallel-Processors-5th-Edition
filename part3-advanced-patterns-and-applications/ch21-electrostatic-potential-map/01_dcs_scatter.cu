// Chapter 21: Electrostatic potential map
// §21.2, Fig. 21.4-21.5: Direct Coulomb Summation (DCS), scatter kernel.
//
// The chapter's base problem (§21.1, Fig. 21.2): for a 2D slice (fixed z)
// of a regular energy grid, potential[gy][gx] = sum over all atoms i of
// atoms[i].charge / distance(gridpoint, atoms[i]).
//
// Fig. 21.3 is the unoptimized C form: for(y) for(x) for(atom) accumulate.
// Fig. 21.4 loop-interchanges this so the OUTER loop runs over atoms and
// the two inner loops (y then x) scatter that atom's contribution to every
// grid point. The interchange is valid because all N_atoms*N_gridpoints
// iterations are independent, and it lets the per-plane z-term and the
// per-row y-term be hoisted out of the two inner loops (computed once per
// atom / once per atom-row instead of once per atom-gridpoint pair):
//   dz is the same for every point in the (single) slice      -> hoist to
//     outside both inner loops (once per atom)
//   dy is the same for every point in a row                   -> hoist to
//     outside the x loop (once per atom-row)
// Fig. 21.5 parallelizes Fig. 21.4 directly: each CUDA thread is one
// iteration of the outermost (atom) loop, i.e. one thread per atom, and
// that thread scatters its atom's contribution into every grid point via
// the same hoisted dz/dy structure. Because many threads (many atoms) can
// target the same grid point concurrently, the grid update (line 17-18 of
// Fig. 21.5) must be an atomic add -- this is exactly the scatter/atomics
// cost the chapter calls out as the downside of this approach (§21.2:
// "this scatter approach ... requires atomic operations ... which
// significantly reduces the speed of parallel execution").
//
// §21.2's host-side design: atoms live in CPU memory, are pushed to GPU
// constant memory in CHUNK_SIZE-atom chunks (so the chunk's byte size fits
// the constant memory budget), and the kernel is invoked once per chunk,
// accumulating into the same device energy grid across chunks.
//
// This file processes a single 2D slice (z = Z_SLICE) of a 3D grid, since
// that is the unit of work Fig. 21.3-21.5's C/kernel code operates on; the
// chapter says the host calls this per-slice function "repeatedly for all
// the slices of the modeled space", which a single-slice sample already
// exercises.

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
#define CHUNK_SIZE 1500  // atoms/chunk; CHUNK_SIZE*4 floats must fit const mem (§21.2)

__constant__ float c_atoms[CHUNK_SIZE * 4];  // packed x,y,z,charge per atom

// ---------------------------------------------------------------------------
// Fig. 21.5: one thread per atom in the current chunk; each thread scatters
// its atom's contribution to every grid point in the slice via atomicAdd.
// ---------------------------------------------------------------------------
__global__ void dcsScatterKernel(int numAtomsInChunk, float *energygrid) {
    int atomIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (atomIdx >= numAtomsInChunk) return;

    float ax = c_atoms[atomIdx * 4 + 0];
    float ay = c_atoms[atomIdx * 4 + 1];
    float az = c_atoms[atomIdx * 4 + 2];
    float charge = c_atoms[atomIdx * 4 + 3];

    // dz is identical for every grid point in this slice (Fig. 21.4 lines
    // 06-07): compute once per atom, outside both inner loops.
    float dz = Z_SLICE - az;
    float dz2 = dz * dz;

    for (int y = 0; y < GRID_Y; ++y) {
        // dy is identical for every grid point in this row (Fig. 21.4
        // lines 11-12): compute once per atom-row, outside the x loop.
        float dy = y * GRIDSPACING - ay;
        float dy2dz2 = dy * dy + dz2;
        for (int x = 0; x < GRID_X; ++x) {
            float dx = x * GRIDSPACING - ax;
            float r = sqrtf(dx * dx + dy2dz2);
            atomicAdd(&energygrid[y * GRID_X + x], charge / r);
        }
    }
}

// ---------------------------------------------------------------------------
// CPU reference: Fig. 21.3's unoptimized triple loop, computed independently
// of any chunking/loop-order choice made on the GPU side.
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

// ---------------------------------------------------------------------------
// Synthetic atoms: x,y span the grid's footprint, z is kept well clear of
// the z=0 slice (|z| >= 3.0) so no atom-gridpoint distance is ever close to
// zero (avoids a near-singular 1/r blowing up the comparison).
// ---------------------------------------------------------------------------
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
        atoms[i * 4 + 2] = 3.0f + nextRand() * 6.0f;      // z in [3, 9]
        atoms[i * 4 + 3] = -2.0f + nextRand() * 4.0f;     // charge in [-2, 2]
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
    printf("DCS scatter kernel (§21.2, Fig. 21.4-21.5):\n");
    printf("grid=%dx%d spacing=%.2f atoms=%d chunk=%d\n", GRID_X, GRID_Y, GRIDSPACING, NUM_ATOMS, CHUNK_SIZE);

    std::vector<float> atoms = makeAtoms(1u);
    std::vector<float> gridRef;
    cpuDcsReference(atoms, gridRef);

    float *grid_d;
    size_t gridBytes = static_cast<size_t>(GRID_X) * GRID_Y * sizeof(float);
    CUDA_CHECK(cudaMalloc(&grid_d, gridBytes));

    int numChunks = (NUM_ATOMS + CHUNK_SIZE - 1) / CHUNK_SIZE;
    const int THREADS = 256;

    auto launch = [&]() {
        CUDA_CHECK(cudaMemset(grid_d, 0, gridBytes));
        for (int c = 0; c < numChunks; ++c) {
            int chunkStart = c * CHUNK_SIZE;
            int chunkAtoms = std::min(CHUNK_SIZE, NUM_ATOMS - chunkStart);
            CUDA_CHECK(cudaMemcpyToSymbol(c_atoms, atoms.data() + static_cast<size_t>(chunkStart) * 4,
                                           static_cast<size_t>(chunkAtoms) * 4 * sizeof(float)));
            int blocks = (chunkAtoms + THREADS - 1) / THREADS;
            dcsScatterKernel<<<blocks, THREADS>>>(chunkAtoms, grid_d);
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
