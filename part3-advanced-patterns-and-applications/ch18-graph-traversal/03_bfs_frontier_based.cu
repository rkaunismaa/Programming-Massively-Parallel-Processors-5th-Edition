// Chapter 18: Graph traversal
// §18.5, Fig. 18.12 & Fig. 18.14: Vertex-centric push BFS kernel with
// frontiers.
//
// The vertex-centric and edge-centric kernels (01/02_bfs_*.cu) re-check
// every vertex/edge at every level, most of which discover they are
// irrelevant and do no useful work. This file instead maintains an
// explicit FRONTIER: the set of vertices discovered at the previous level.
// Each level, a thread is launched only for each element of the previous
// frontier (not for every vertex in the graph), walks that vertex's
// outgoing edges, and any thread that succeeds in visiting an unvisited
// neighbor appends it to the current frontier via an atomic increment of
// the frontier's counter. This reduces the work to the ideal O(n + m)
// (§18.5): every vertex is checked once and every edge is traversed once.
//
// Because multiple threads can now observe the same unvisited neighbor
// concurrently and each of them would otherwise append it to the frontier
// redundantly, the "is it unvisited, and if so label it" check must happen
// as a single atomic operation -- unlike the benign, unsynchronized race
// used in 01_bfs_vertex_centric.cu. Fig. 18.14's visitVertexAtomically uses
// an atomic compare-and-swap (cuda::atomic_ref::compare_exchange_strong) to
// do exactly this: compare level[vertex] against UNVISITED and, if equal,
// swap in currLevel, atomically.

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cuda/atomic>
#include <queue>
#include <vector>

#include "../../common/cuda_utils.h"

const int UNVISITED = INT_MAX;

// ---------------------------------------------------------------------------
// §18.5, Fig. 18.14: atomically check-and-label a vertex using
// compare-and-swap so exactly one thread observes success even if many
// threads race on the same unvisited neighbor.
// ---------------------------------------------------------------------------
__device__ bool visitVertexAtomically(int *level, int vertex, int currLevel) {
    cuda::atomic_ref<int, cuda::thread_scope_device> levelRef(level[vertex]);
    int unvisited = UNVISITED;
    return levelRef.compare_exchange_strong(unvisited, currLevel, cuda::memory_order_relaxed, cuda::memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// §18.5, Fig. 18.12: vertex-centric push BFS kernel with frontiers -- one
// thread per element of the PREVIOUS frontier, called once per level.
// ---------------------------------------------------------------------------
__global__ void bfsFrontierKernel(const int *srcPtrs, const int *dst, int *level, int currLevel, const int *prevFrontier, int numPrevFrontier, int *currFrontier, int *numCurrFrontier) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numPrevFrontier) {
        int vertex = prevFrontier[i];
        for (int edge = srcPtrs[vertex]; edge < srcPtrs[vertex + 1]; ++edge) {
            int neighbor = dst[edge];
            if (visitVertexAtomically(level, neighbor, currLevel)) {
                int currFrontierIdx = atomicAdd(numCurrFrontier, 1);
                currFrontier[currFrontierIdx] = neighbor;
            }
        }
    }
}

struct Edge {
    int src, dst;
};

// Same small graph as 01/02_bfs_*.cu (13 vertices, root 0, vertex 12
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

float runBfsFrontier(const std::vector<int> &srcPtrs_h, const std::vector<int> &dst_h, int numVertices, int root, std::vector<int> &level_h) {
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

    int blockDim = 256;

    auto runOnce = [&]() {
        std::vector<int> initLevel(numVertices, UNVISITED);
        initLevel[root] = 0;
        CUDA_CHECK(cudaMemcpy(level_d, initLevel.data(), numVertices * sizeof(int), cudaMemcpyHostToDevice));

        int *prevFrontier_d = frontierA_d;
        int *currFrontier_d = frontierB_d;
        int rootVertex = root;
        CUDA_CHECK(cudaMemcpy(prevFrontier_d, &rootVertex, sizeof(int), cudaMemcpyHostToDevice));

        int numPrevFrontier = 1;
        int currLevel = 1;
        while (numPrevFrontier > 0) {
            CUDA_CHECK(cudaMemset(numCurrFrontier_d, 0, sizeof(int)));
            int gridDim = (numPrevFrontier + blockDim - 1) / blockDim;
            bfsFrontierKernel<<<gridDim, blockDim>>>(srcPtrs_d, dst_d, level_d, currLevel, prevFrontier_d, numPrevFrontier, currFrontier_d, numCurrFrontier_d);
            CUDA_CHECK(cudaGetLastError());

            int numCurrFrontier_h;
            CUDA_CHECK(cudaMemcpy(&numCurrFrontier_h, numCurrFrontier_d, sizeof(int), cudaMemcpyDeviceToHost));

            int *tmp = prevFrontier_d;
            prevFrontier_d = currFrontier_d;
            currFrontier_d = tmp;
            numPrevFrontier = numCurrFrontier_h;
            ++currLevel;
        }
    };

    runOnce();
    CUDA_CHECK(cudaDeviceSynchronize());

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
    float ms = runBfsFrontier(srcPtrs, dst, numVertices, root, level);

    bool ok = checkExact(level, ref);
    printf("%s (V=%d, E=%zu): %.4f ms  [%s]\n", name, numVertices, edges.size(), ms, ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    printf("BFS: vertex-centric push with frontiers, work-efficient O(n+m) (§18.5, Fig. 18.12/18.14):\n");
    bool ok = true;
    ok = runCase("small graph (root=0)", 13, 0, smallGraphEdges()) && ok;
    ok = runCase("random 4000-vertex graph", 4000, 0, randomGraphEdges(4000, 6, 123u)) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
