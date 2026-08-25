// Chapter 21: Electrostatic potential map
// §21.5, Fig. 21.11-21.14: cutoff binning for data-size scalability.
//
// §21.5 replaces exact DCS (every grid point sums over every atom, O(N*M))
// with cutoff summation: each grid point sums only atoms within a fixed
// cutoff radius, trading a small amount of accuracy for O(N) scaling with
// system volume. The chapter rules out an atom-centric (scatter) cutoff
// kernel for the same reason §21.2 ruled out atom-centric DCS -- scatter
// needs atomics -- and instead builds a grid-centric cutoff *binning*
// algorithm (credited to Rodrigues et al. [3]):
//
//   - atoms are sorted once into spatial bins, "implemented as
//     multi-dimensional arrays: the x, y, and z dimensions [of bins] ...
//     and a vector of atoms in the bin" (this file bins by x,y only, since
//     -- like every other file in this chapter -- it operates on one fixed-
//     z 2D slice; see the header note on this below);
//   - "based on the block dimensions and the grid spacing, one can
//     calculate the area ... covered by each block" and Fig. 21.13 assumes
//     "each of these areas is also covered by a bin" -- i.e. bin size =
//     one thread block's grid-point footprint. This file uses that literal
//     correspondence: BIN_DIM x BIN_DIM grid points per bin, one CUDA
//     block per bin, so blockIdx == bin coordinate directly;
//   - for a given bin, its "neighborhood" is the fixed list of relative
//     bin offsets that could contain an atom within cutoff distance of any
//     grid point the block owns (Fig. 21.12/21.14). The chapter computes
//     this with a conservative "super circle" (bin-center distance <=
//     cutoff + half the bin diagonal); this file instead computes the
//     exact axis-aligned-box-to-box minimum distance between the origin
//     bin and each candidate neighbor bin and keeps it iff that minimum
//     distance <= cutoff. This is a documented, deliberate substitution:
//     it is the precise version of the same "is this bin worth
//     examining" test the chapter's approximation exists to answer, gives
//     the exact same kind of single fixed list per block ("prepared before
//     launching the grid ... supplied to the kernel ... as a constant
//     memory array", §21.5), and -- being exact rather than a conservative
//     superset -- cannot itself introduce error into the cutoff-restricted
//     result (the kernel's own per-atom distance check is still the sole
//     source of the cutoff decision);
//   - within a neighborhood bin, "threads in a block collaborate in
//     loading the atom information ... into shared memory. All threads
//     then examine the atoms out of shared memory," each independently
//     deciding inclusion/exclusion by distance -- exactly what the kernel
//     below does, chunking each bin's atoms through a blockDim-sized
//     shared-memory staging buffer;
//   - atoms live in global memory, not constant memory, because "thread
//     blocks will be accessing different neighborhoods [so] the limited
//     sized constant memory will unlikely be able to hold all the atoms
//     that are needed by all active thread blocks" (§21.5).
//
// Deliberately NOT implemented (out of scope for this sample): fixed-
// capacity bins with a host-side overflow list for atoms that don't fit
// (§21.5's further memory-coalescing refinement for bin storage). This
// file instead uses a variable-length CSR bin layout (binStart/binCount
// into a single atoms-sorted-by-bin array), which loses that specific
// coalescing optimization but cannot drop any atom, so no overflow
// handling is needed for correctness.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#include "../../common/cuda_utils.h"

#define GRID_X 96
#define GRID_Y 96
#define GRIDSPACING 0.5f
#define Z_SLICE 0.0f
#define BIN_DIM 8  // grid points per bin per axis == one thread block's tile (Fig. 21.13)
#define CUTOFF 12.0f  // matches the chapter's own worked example ("a typical cut-off
                       // distance of 12 A for molecular-level force calculation")

#define NUM_ATOMS 4000
#define MAX_NEIGHBORS 128

#define NUM_BINS_X (GRID_X / BIN_DIM)
#define NUM_BINS_Y (GRID_Y / BIN_DIM)
#define BIN_SIZE_X (BIN_DIM * GRIDSPACING)
#define BIN_SIZE_Y (BIN_DIM * GRIDSPACING)

__constant__ int2 c_neighborOffsets[MAX_NEIGHBORS];
__constant__ int c_numNeighbors;

// ---------------------------------------------------------------------------
// §21.5, Fig. 21.12-21.14: one CUDA block per bin (blockIdx.{x,y} IS the
// bin coordinate, by construction), one thread per grid point within that
// bin's BIN_DIM x BIN_DIM tile. For each neighborhood bin (from the fixed
// per-kernel offset list in constant memory), the block cooperatively
// stages that bin's atoms through shared memory in blockDim-sized chunks;
// each thread then independently checks every staged atom against its own
// cutoff distance before accumulating.
// ---------------------------------------------------------------------------
__global__ void dcsCutoffBinningKernel(const float4 *atomsSorted, const int *binStart, const int *binCount,
                                        float *energygrid) {
    extern __shared__ float4 sh_atoms[];

    int gx = blockIdx.x * BIN_DIM + threadIdx.x;
    int gy = blockIdx.y * BIN_DIM + threadIdx.y;
    float x = gx * GRIDSPACING;
    float y = gy * GRIDSPACING;
    float cutoff2 = CUTOFF * CUTOFF;

    int chunkCap = blockDim.x * blockDim.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    float energy = 0.0f;
    for (int n = 0; n < c_numNeighbors; ++n) {
        int nbx = blockIdx.x + c_neighborOffsets[n].x;
        int nby = blockIdx.y + c_neighborOffsets[n].y;
        if (nbx < 0 || nbx >= NUM_BINS_X || nby < 0 || nby >= NUM_BINS_Y) continue;

        int binIdx = nby * NUM_BINS_X + nbx;
        int start = binStart[binIdx];
        int count = binCount[binIdx];

        for (int chunkStart = 0; chunkStart < count; chunkStart += chunkCap) {
            int idxInBin = chunkStart + tid;
            if (idxInBin < count) sh_atoms[tid] = atomsSorted[start + idxInBin];
            __syncthreads();

            int numInChunk = min(chunkCap, count - chunkStart);
            for (int k = 0; k < numInChunk; ++k) {
                float4 a = sh_atoms[k];
                float dx = x - a.x, dy = y - a.y, dz = Z_SLICE - a.z;
                float r2 = dx * dx + dy * dy + dz * dz;
                if (r2 <= cutoff2) energy += a.w / sqrtf(r2);
            }
            __syncthreads();
        }
    }

    if (gx < GRID_X && gy < GRID_Y) energygrid[gy * GRID_X + gx] = energy;
}

// ---------------------------------------------------------------------------
// CPU reference: direct cutoff summation (no binning), restricted to the
// same CUTOFF radius, computed independently of the GPU's bin/neighborhood
// machinery.
// ---------------------------------------------------------------------------
static void cpuCutoffReference(const std::vector<float> &atoms, std::vector<float> &grid) {
    grid.assign(static_cast<size_t>(GRID_X) * GRID_Y, 0.0f);
    double cutoff2 = static_cast<double>(CUTOFF) * CUTOFF;
    for (int y = 0; y < GRID_Y; ++y) {
        float gy = y * GRIDSPACING;
        for (int x = 0; x < GRID_X; ++x) {
            float gx = x * GRIDSPACING;
            double energy = 0.0;
            for (int i = 0; i < NUM_ATOMS; ++i) {
                float ax = atoms[i * 4 + 0], ay = atoms[i * 4 + 1];
                float az = atoms[i * 4 + 2], charge = atoms[i * 4 + 3];
                double dx = gx - ax, dy = gy - ay, dz = Z_SLICE - az;
                double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 <= cutoff2) energy += static_cast<double>(charge) / std::sqrt(r2);
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

// Minimum distance between two axis-aligned boxes along one axis, given the
// box origins (lo) and a shared box length (len).
static float axisGap(float loA, float loB, float len) {
    if (loB >= loA + len) return loB - (loA + len);
    if (loA >= loB + len) return loA - (loB + len);
    return 0.0f;
}

// Builds the fixed neighborhood offset list (§21.5, Fig. 21.14): all
// relative bin offsets (di, dj) whose bin could contain a point within
// CUTOFF of some point in the origin bin, via exact box-to-box minimum
// distance (see file header for why this replaces the book's "super
// circle" conservative approximation).
static std::vector<int2> buildNeighborOffsets() {
    std::vector<int2> offsets;
    float halfDiag = 0.5f * std::sqrt(BIN_SIZE_X * BIN_SIZE_X + BIN_SIZE_Y * BIN_SIZE_Y);
    int K = static_cast<int>(std::ceil((CUTOFF + halfDiag) / std::min(BIN_SIZE_X, BIN_SIZE_Y))) + 1;
    for (int dj = -K; dj <= K; ++dj) {
        for (int di = -K; di <= K; ++di) {
            float gapX = axisGap(0.0f, di * BIN_SIZE_X, BIN_SIZE_X);
            float gapY = axisGap(0.0f, dj * BIN_SIZE_Y, BIN_SIZE_Y);
            float minDist = std::sqrt(gapX * gapX + gapY * gapY);
            if (minDist <= CUTOFF) {
                int2 o;
                o.x = di;
                o.y = dj;
                offsets.push_back(o);
            }
        }
    }
    return offsets;
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
    static_assert(GRID_X % BIN_DIM == 0, "GRID_X must be a multiple of BIN_DIM");
    static_assert(GRID_Y % BIN_DIM == 0, "GRID_Y must be a multiple of BIN_DIM");
    printf("DCS cutoff binning kernel (§21.5, Fig. 21.11-21.14):\n");
    printf("grid=%dx%d spacing=%.2f atoms=%d bin=%dx%d (%d bins) cutoff=%.1f\n", GRID_X, GRID_Y, GRIDSPACING,
           NUM_ATOMS, BIN_DIM, BIN_DIM, NUM_BINS_X * NUM_BINS_Y, CUTOFF);

    std::vector<float> atoms = makeAtoms(1u);
    std::vector<float> gridRef;
    cpuCutoffReference(atoms, gridRef);

    // Host-side counting sort of atoms into bins (CSR layout: binStart/binCount
    // index into atomsSortedByBin).
    int numBins = NUM_BINS_X * NUM_BINS_Y;
    std::vector<int> binCount(numBins, 0);
    std::vector<int> atomBin(NUM_ATOMS);
    for (int i = 0; i < NUM_ATOMS; ++i) {
        int bx = std::min(static_cast<int>(atoms[i * 4 + 0] / BIN_SIZE_X), NUM_BINS_X - 1);
        int by = std::min(static_cast<int>(atoms[i * 4 + 1] / BIN_SIZE_Y), NUM_BINS_Y - 1);
        int bin = by * NUM_BINS_X + bx;
        atomBin[i] = bin;
        binCount[bin]++;
    }
    std::vector<int> binStart(numBins);
    int running = 0;
    for (int b = 0; b < numBins; ++b) {
        binStart[b] = running;
        running += binCount[b];
    }
    std::vector<float4> atomsSorted(NUM_ATOMS);
    std::vector<int> fillCursor = binStart;
    for (int i = 0; i < NUM_ATOMS; ++i) {
        int bin = atomBin[i];
        int dst = fillCursor[bin]++;
        atomsSorted[dst] = make_float4(atoms[i * 4 + 0], atoms[i * 4 + 1], atoms[i * 4 + 2], atoms[i * 4 + 3]);
    }

    std::vector<int2> neighborOffsets = buildNeighborOffsets();
    if (neighborOffsets.size() > MAX_NEIGHBORS) {
        fprintf(stderr, "neighborhood list (%zu) exceeds MAX_NEIGHBORS (%d)\n", neighborOffsets.size(),
                MAX_NEIGHBORS);
        return EXIT_FAILURE;
    }
    printf("neighborhood list size=%zu bins/block\n", neighborOffsets.size());

    float4 *atomsSorted_d;
    int *binStart_d, *binCount_d;
    float *grid_d;
    CUDA_CHECK(cudaMalloc(&atomsSorted_d, NUM_ATOMS * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&binStart_d, numBins * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&binCount_d, numBins * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&grid_d, static_cast<size_t>(GRID_X) * GRID_Y * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(atomsSorted_d, atomsSorted.data(), NUM_ATOMS * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(binStart_d, binStart.data(), numBins * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(binCount_d, binCount.data(), numBins * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpyToSymbol(c_neighborOffsets, neighborOffsets.data(), neighborOffsets.size() * sizeof(int2)));
    int numNeighbors = static_cast<int>(neighborOffsets.size());
    CUDA_CHECK(cudaMemcpyToSymbol(c_numNeighbors, &numNeighbors, sizeof(int)));

    dim3 block(BIN_DIM, BIN_DIM);
    dim3 grid(NUM_BINS_X, NUM_BINS_Y);
    size_t shmemBytes = static_cast<size_t>(BIN_DIM) * BIN_DIM * sizeof(float4);

    auto launch = [&]() {
        dcsCutoffBinningKernel<<<grid, block, shmemBytes>>>(atomsSorted_d, binStart_d, binCount_d, grid_d);
        CUDA_CHECK(cudaGetLastError());
    };

    launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    launch();
    float ms = timer.stopAndGetMs();

    std::vector<float> gridOut(static_cast<size_t>(GRID_X) * GRID_Y);
    CUDA_CHECK(cudaMemcpy(gridOut.data(), grid_d, gridOut.size() * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(atomsSorted_d));
    CUDA_CHECK(cudaFree(binStart_d));
    CUDA_CHECK(cudaFree(binCount_d));
    CUDA_CHECK(cudaFree(grid_d));

    float maxDiff = 0.0f;
    bool ok = checkClose(gridOut, gridRef, 1e-3f, &maxDiff);
    printf("GPU vs CPU cutoff reference: max|diff|=%.6f  %.4f ms  [%s]\n", maxDiff, ms, ok ? "match" : "MISMATCH");

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
