// Chapter 18: Graph traversal
// §18.7, Fig. 18.17: Multi-level BFS kernel using cooperative groups.
//
// Every previous kernel in this chapter is launched once PER LEVEL from
// the host, because each level's kernel must fully finish (all previous-
// level vertices labeled) before the next level's kernel can safely start.
// When frontiers are small (low-average-degree graphs like road networks),
// the overhead of tearing down and relaunching a grid for every level can
// dominate the actual work done. This file instead launches a SINGLE grid
// that loops over all BFS levels internally, using the cooperative-groups
// API's grid.sync() to perform a grid-wide barrier between levels instead
// of a kernel relaunch.
//
// A grid-wide barrier is only safe if every thread block in the grid is
// guaranteed to be resident on an SM at the same time (otherwise blocks
// waiting to be scheduled could deadlock against blocks waiting at the
// barrier for them). So, per §18.7, the host computes the maximum number
// of blocks that can run concurrently via
// cudaOccupancyMaxActiveBlocksPerMultiprocessor() * (SM count) and launches
// exactly that many blocks -- no more -- via cudaLaunchCooperativeKernel(),
// which requires kernel arguments to be packed into a void* array rather
// than passed via the usual <<<...>>> syntax. Because the grid may now have
// fewer threads than there are frontier vertices, each thread processes
// multiple frontier elements in a grid-stride loop (grid.thread_rank(),
// grid.num_threads()) instead of the "one thread per frontier element"
// mapping used in 03/04_bfs_frontier*.cu.
//
// cudaDeviceGetAttribute(cudaDevAttrCooperativeLaunch, ...) is checked
// first and the program exits 0 with a clear skip message (not a FAIL) if
// the device doesn't support cooperative grid launch -- defensive code for
// machines other than the two used to build this repo, both of which do
// support it.

#include <climits>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cuda/atomic>
#include <queue>
#include <vector>

#include "../../common/cuda_utils.h"

namespace cg = cooperative_groups;

const int UNVISITED = INT_MAX;

// ---------------------------------------------------------------------------
// §18.5, Fig. 18.14: same atomic check-and-label device function used by
// the other frontier-based kernels.
// ---------------------------------------------------------------------------
__device__ bool visitVertexAtomically(int *level, int vertex, int currLevel) {
    cuda::atomic_ref<int, cuda::thread_scope_device> levelRef(level[vertex]);
    int unvisited = UNVISITED;
    return levelRef.compare_exchange_strong(unvisited, currLevel, cuda::memory_order_relaxed, cuda::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// §18.7, Fig. 18.17: single-launch, multi-level BFS kernel. Iterates over
// levels internally, using grid.sync() as the barrier between levels
// instead of a host-side kernel relaunch.
//
// numPrevFrontier is passed BY VALUE (every thread starts with the same
// initial frontier size and keeps its own up-to-date copy in a register,
// re-read from *numCurrFrontier after each grid.sync()); numCurrFrontier is
// the one counter that actually lives in device memory, since threads
// across the whole grid atomically add to it.
// ---------------------------------------------------------------------------
__global__ void bfsCoopKernel(const int *srcPtrs, const int *dst, int *level, int *prevFrontier, int *currFrontier, int numPrevFrontier, int *numCurrFrontier) {
    cg::grid_group grid = cg::this_grid();
    int currLevel = 1;

    while (numPrevFrontier > 0) {
        for (int i = grid.thread_rank(); i < numPrevFrontier; i += grid.num_threads()) {
            int vertex = prevFrontier[i];
            for (int edge = srcPtrs[vertex]; edge < srcPtrs[vertex + 1]; ++edge) {
                int neighbor = dst[edge];
                if (visitVertexAtomically(level, neighbor, currLevel)) {
                    int currFrontierIdx = atomicAdd(numCurrFrontier, 1);
                    currFrontier[currFrontierIdx] = neighbor;
                }
            }
        }

        grid.sync();
        numPrevFrontier = *numCurrFrontier;
        grid.sync();
        if (grid.thread_rank() == 0) {
            *numCurrFrontier = 0;
        }
        grid.sync();

        int *tmp = prevFrontier;
        prevFrontier = currFrontier;
        currFrontier = tmp;
        ++currLevel;
    }
}

struct Edge {
    int src, dst;
};

// Same small graph as 01/02/03/04_bfs_*.cu (13 vertices, root 0, vertex 12
// unreachable).
std::vector<Edge> smallGraphEdges() {
    return {
        {0, 1}, {0, 2}, {1, 3}, {1, 4}, {2, 4}, {2, 5}, {3, 6}, {4, 6}, {4, 7}, {5, 7}, {5, 8}, {6, 9}, {7, 9}, {7, 10}, {8, 10}, {9, 11}, {10, 11}};
}

std::vector<Edge> randomGraphEdges(int numVertices, int avgOutDegree, unsigned int seed) {
    std::vector<Edge> edges;
    unsigned int state = seed;
    auto nextRand = [&]() -> unsigned int {
        state = state * 1103515245u + 12345u;
        return (state >> 8) & 0xFFFFFFu;
    };
    for (int v = 0; v < numVertices; ++v) {
        int degree = 1 + static_cast<int>(nextRand() % (2u * avgOutDegree));
        for (int k = 0; k < degree; ++k) {
            int d = static_cast<int>(nextRand() % numVertices);
            if (d != v) edges.push_back({v, d});
        }
    }
    return edges;
}

void buildCsr(const std::vector<Edge> &edges, int numVertices, std::vector<int> &srcPtrs, std::vector<int> &dst) {
    srcPtrs.assign(numVertices + 1, 0);
    for (const auto &e : edges) srcPtrs[e.src + 1]++;
    for (int v = 0; v < numVertices; ++v) srcPtrs[v + 1] += srcPtrs[v];
    dst.assign(edges.size(), 0);
    std::vector<int> cursor(srcPtrs.begin(), srcPtrs.end() - 1);
    for (const auto &e : edges) dst[cursor[e.src]++] = e.dst;
}

std::vector<int> cpuBfs(const std::vector<int> &srcPtrs, const std::vector<int> &dst, int numVertices, int root) {
    std::vector<int> level(numVertices, UNVISITED);
    level[root] = 0;
    std::queue<int> q;
    q.push(root);
    while (!q.empty()) {
        int v = q.front();
        q.pop();
        for (int e = srcPtrs[v]; e < srcPtrs[v + 1]; ++e) {
            int nb = dst[e];
            if (level[nb] == UNVISITED) {
                level[nb] = level[v] + 1;
                q.push(nb);
            }
        }
    }
    return level;
}

// §18.7: compute the max number of blocks that can run concurrently on this
// device for this kernel/block size, via the occupancy calculator times the
// SM count -- the grid must not exceed this or grid.sync() could deadlock.
unsigned int maxCooperativeBlocks(int numThreadsPerBlock) {
    int numBlocksPerSM = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM, bfsCoopKernel, numThreadsPerBlock, 0));
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));
    int numSMs = deviceProp.multiProcessorCount;
    return static_cast<unsigned int>(numSMs * numBlocksPerSM);
}

float runBfsCoop(const std::vector<int> &srcPtrs_h, const std::vector<int> &dst_h, int numVertices, int root, std::vector<int> &level_h) {
    int numEdges = static_cast<int>(dst_h.size());

    int *srcPtrs_d, *dst_d, *level_d;
    int *frontierA_d, *frontierB_d, *numCurrFrontier_d;
    CUDA_CHECK(cudaMalloc(&srcPtrs_d, (numVertices + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dst_d, numEdges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&level_d, numVertices * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&frontierA_d, numVertices * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&frontierB_d, numVertices * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&numCurrFrontier_d, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(srcPtrs_d, srcPtrs_h.data(), (numVertices + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dst_d, dst_h.data(), numEdges * sizeof(int), cudaMemcpyHostToDevice));

    int numThreadsPerBlock = 256;
    unsigned int numBlocks = maxCooperativeBlocks(numThreadsPerBlock);
    if (numBlocks == 0) numBlocks = 1;

    auto runOnce = [&]() {
        std::vector<int> initLevel(numVertices, UNVISITED);
        initLevel[root] = 0;
        CUDA_CHECK(cudaMemcpy(level_d, initLevel.data(), numVertices * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(numCurrFrontier_d, 0, sizeof(int)));

        int rootVertex = root;
        CUDA_CHECK(cudaMemcpy(frontierA_d, &rootVertex, sizeof(int), cudaMemcpyHostToDevice));

        int numPrevFrontier = 1;
        // §18.7, Fig. 18.17: pack kernel arguments into a void* array and
        // launch via cudaLaunchCooperativeKernel instead of <<<...>>>.
        void *kernelArgs[] = {(void *)&srcPtrs_d, (void *)&dst_d, (void *)&level_d, (void *)&frontierA_d, (void *)&frontierB_d, (void *)&numPrevFrontier, (void *)&numCurrFrontier_d};
        CUDA_CHECK(cudaLaunchCooperativeKernel((void *)bfsCoopKernel, numBlocks, numThreadsPerBlock, kernelArgs));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    };

    runOnce();

    GpuTimer timer;
    timer.start();
    runOnce();
    float ms = timer.stopAndGetMs();

    level_h.resize(numVertices);
    CUDA_CHECK(cudaMemcpy(level_h.data(), level_d, numVertices * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(srcPtrs_d));
    CUDA_CHECK(cudaFree(dst_d));
    CUDA_CHECK(cudaFree(level_d));
    CUDA_CHECK(cudaFree(frontierA_d));
    CUDA_CHECK(cudaFree(frontierB_d));
    CUDA_CHECK(cudaFree(numCurrFrontier_d));
    return ms;
}

bool checkExact(const std::vector<int> &a, const std::vector<int> &b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) return false;
    return true;
}

bool runCase(const char *name, int numVertices, int root, const std::vector<Edge> &edges) {
    std::vector<int> srcPtrs, dst;
    buildCsr(edges, numVertices, srcPtrs, dst);
    std::vector<int> ref = cpuBfs(srcPtrs, dst, numVertices, root);

    std::vector<int> level;
    float ms = runBfsCoop(srcPtrs, dst, numVertices, root, level);

    bool ok = checkExact(level, ref);
    printf("%s (V=%d, E=%zu): %.4f ms  [%s]\n", name, numVertices, edges.size(), ms, ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    printf("BFS: single-launch multi-level kernel with cooperative-groups grid sync (§18.7, Fig. 18.17):\n");

    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    int supportsCoopLaunch = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, device));
    if (!supportsCoopLaunch) {
        printf("Device %d does not support cudaLaunchCooperativeKernel (cudaDevAttrCooperativeLaunch=0); "
               "skipping this sample -- grid-wide cooperative-groups sync is unavailable on this GPU.\n",
               device);
        printf("PASS (skipped: cooperative launch unsupported on this device)\n");
        return 0;
    }

    bool ok = true;
    ok = runCase("small graph (root=0)", 13, 0, smallGraphEdges()) && ok;
    ok = runCase("random 4000-vertex graph", 4000, 0, randomGraphEdges(4000, 6, 123u)) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
