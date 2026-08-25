// Chapter 16: Dynamic programming and wavefront parallelism
// §16.4, Fig. 16.4: bottom-up parallel Floyd-Warshall all-pairs shortest path.
//
// d(i, j, k) = shortest distance from i to j using only vertices 0..k as
// intermediate vertices. The recurrence (§16.4):
//   d(i, j, k) = min(d(i, j, k-1), d(i, k, k-1) + d(k, j, k-1))
// is computed bottom-up: the host launches one kernel per value of k
// (Fig. 16.4's host code), and within an iteration every (i, j) cell of the
// distance table can be updated in parallel (constant-size wavefront, per
// §16.3's discussion of Floyd-Warshall's wavefront pattern being one 2D
// plane of a 3D (i, j, k) problem space per iteration). Kernel termination
// + relaunch is the global synchronization between iterations k -> k+1.
//
// FW_bottomup below reproduces Fig. 16.4's kernel line for line: one thread
// per (row, col) cell of the distance table, thread block covers a row
// section (dimBlock is 1D, dimGrid is 2D: x covers column sections, y is
// exactly n_vertices so each block row maps to one distance-table row).
// Thread 0 of each block stages dist[row][k] into shared memory
// (dist_k_col); every thread then reads dist[k][col] from global memory
// (coalesced across the block) and updates its own cell if routing through
// k is shorter.
//
// Book note: the figure names the "no edge" sentinel INFINITY. This file
// renames it to INF because <cmath> (pulled in by cuda_utils.h) already
// defines INFINITY as a float macro; the renaming is purely cosmetic, the
// kernel logic is unchanged.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>

#include "../../common/cuda_utils.h"

#define INF (1 << 20)  // sentinel for "no edge" / unreachable

// ---------------------------------------------------------------------------
// §16.4, Fig. 16.4 (kernel): one thread updates one dist[row][col] cell for
// the current intermediate vertex k.
// ---------------------------------------------------------------------------
__global__ void FW_bottomup(int k, int *dist, int n_vertices) {
    // Distance table column and row
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n_vertices) return;  // note: threads that exit here never reach line 54's __syncthreads() below; correct on sm_75 (verbatim from the book) but formally relies on exited threads being excluded from the barrier rather than each live thread executing it uniformly.
    int row = blockIdx.y;

    // Index in dist matrix
    int distIndex = n_vertices * row + col;

    __shared__ int dist_k_col;
    // Distance to intermediate vertex with column k
    if (threadIdx.x == 0)
        dist_k_col = dist[n_vertices * row + k];
    __syncthreads();
    if (dist_k_col == INF) {
        return;
    }

    // Distance to intermediate vertex with row k
    int dist_k_row = dist[k * n_vertices + col];
    if (dist_k_row == INF) {
        return;
    }

    // Update if vertex k is on the shortest path
    int new_distance = dist_k_col + dist_k_row;
    if (new_distance < dist[distIndex])
        dist[distIndex] = new_distance;
}

// ---------------------------------------------------------------------------
// CPU reference: textbook triple-nested-loop Floyd-Warshall (same recurrence,
// sequential over k).
// ---------------------------------------------------------------------------
void floydWarshallCPU(std::vector<int> &dist, int n) {
    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < n; ++i) {
            if (dist[i * n + k] == INF) continue;
            for (int j = 0; j < n; ++j) {
                if (dist[k * n + j] == INF) continue;
                int nd = dist[i * n + k] + dist[k * n + j];
                if (nd < dist[i * n + j]) dist[i * n + j] = nd;
            }
        }
    }
}

// Random directed weighted graph: dist[i][i] = 0, dist[i][j] = random weight
// with probability edgeProb, otherwise INF (no edge).
std::vector<int> generateGraph(int n, double edgeProb, unsigned int seed) {
    std::vector<int> dist(static_cast<size_t>(n) * n, INF);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    std::uniform_int_distribution<int> weight(1, 20);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                dist[i * n + j] = 0;
            } else if (coin(rng) < edgeProb) {
                dist[i * n + j] = weight(rng);
            }
        }
    }
    return dist;
}

// ---------------------------------------------------------------------------
// §16.4, Fig. 16.4 (host code): one kernel launch per intermediate vertex k.
// ---------------------------------------------------------------------------
float runFloydWarshallGPU(std::vector<int> &dist_h, int n_vertices, int threads) {
    size_t bytes = static_cast<size_t>(n_vertices) * n_vertices * sizeof(int);
    int *dist_d;
    CUDA_CHECK(cudaMalloc(&dist_d, bytes));
    CUDA_CHECK(cudaMemcpy(dist_d, dist_h.data(), bytes, cudaMemcpyHostToDevice));

    dim3 dimBlock(threads);
    dim3 dimGrid((n_vertices + threads - 1) / threads, n_vertices);

    GpuTimer timer;
    timer.start();
    for (int k = 0; k < n_vertices; ++k) {
        FW_bottomup<<<dimGrid, dimBlock>>>(k, dist_d, n_vertices);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    float ms = timer.stopAndGetMs();

    CUDA_CHECK(cudaMemcpy(dist_h.data(), dist_d, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dist_d));
    return ms;
}

bool runTestCase(int n_vertices, double edgeProb, int threads) {
    std::vector<int> dist_gpu = generateGraph(n_vertices, edgeProb, 1234u + static_cast<unsigned int>(n_vertices));
    std::vector<int> dist_cpu = dist_gpu;

    floydWarshallCPU(dist_cpu, n_vertices);
    float ms = runFloydWarshallGPU(dist_gpu, n_vertices, threads);

    bool ok = (dist_gpu == dist_cpu);
    printf("n_vertices=%d edgeProb=%.2f threads=%d: %.4f ms  [%s]\n", n_vertices, edgeProb, threads, ms,
           ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    printf("Parallel bottom-up Floyd-Warshall all-pairs shortest path (§16.4, Fig. 16.4):\n");
    bool ok = true;
    ok = runTestCase(1, 0.5, 32) && ok;
    ok = runTestCase(5, 0.5, 32) && ok;    // small graph, single block in x
    ok = runTestCase(37, 0.3, 32) && ok;   // not a multiple of block size
    ok = runTestCase(128, 0.15, 64) && ok;
    ok = runTestCase(200, 0.05, 128) && ok;  // sparse, exercises INF short-circuits

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
