// Chapter 18: Graph traversal
// §18.3, Fig. 18.6: Vertex-centric push (top-down) BFS kernel.
//
// One thread per VERTEX. Each level, every thread checks whether its own
// vertex belongs to the previous level; if so, it walks that vertex's
// outgoing edges (via the CSR srcPtrs/dst arrays) and labels each unvisited
// neighbor as belonging to the current level, setting a "new vertex
// visited" flag the host uses to decide whether another level needs to be
// launched. As the text notes, multiple threads may label the same
// neighbor and multiple threads may set the flag -- these are technically
// race conditions, but they are benign because the write is idempotent
// (every racing thread writes the same value), so no atomics are used here
// (§18.3's own caveat: "conservative programmers should use an atomic
// operation for these two operations" -- §18.5 does exactly that once
// frontiers make the write's uniqueness matter).
//
// Work efficiency: every vertex is re-checked at every level, so this is
// O(d*n + m) work (§18.5) rather than the ideal O(n + m) -- fixed by the
// frontier-based approach in 03/04/05_bfs_*.cu.

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <queue>
#include <vector>

#include "../../common/cuda_utils.h"

const int UNVISITED = INT_MAX;

// ---------------------------------------------------------------------------
// §18.3, Fig. 18.6: vertex-centric push BFS kernel -- one thread per vertex,
// called once per level.
// ---------------------------------------------------------------------------
__global__ void bfsPushKernel(const int *srcPtrs, const int *dst, int numVertices, int *level, int currLevel, int *newVertexVisited) {
    int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex < numVertices) {
        if (level[vertex] == currLevel - 1) {
            for (int edge = srcPtrs[vertex]; edge < srcPtrs[vertex + 1]; ++edge) {
                int neighbor = dst[edge];
                if (level[neighbor] == UNVISITED) {
                    level[neighbor] = currLevel;
                    *newVertexVisited = 1;
                }
            }
        }
    }
}

struct Edge {
    int src, dst;
};

// Small graph (13 vertices, root 0) with vertex 12 deliberately left
// unreachable so the UNVISITED sentinel is exercised on both host and
// device. Levels come out as: {0}, {1,2}, {3,4,5}, {6,7,8}, {9,10}, {11}.
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

float runBfsPush(const std::vector<int> &srcPtrs_h, const std::vector<int> &dst_h, int numVertices, int root, std::vector<int> &level_h) {
    int numEdges = static_cast<int>(dst_h.size());

    int *srcPtrs_d, *dst_d, *level_d, *flag_d;
    CUDA_CHECK(cudaMalloc(&srcPtrs_d, (numVertices + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dst_d, numEdges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&level_d, numVertices * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&flag_d, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(srcPtrs_d, srcPtrs_h.data(), (numVertices + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dst_d, dst_h.data(), numEdges * sizeof(int), cudaMemcpyHostToDevice));

    int blockDim = 256;
    int gridDim = (numVertices + blockDim - 1) / blockDim;

    auto runOnce = [&]() {
        std::vector<int> initLevel(numVertices, UNVISITED);
        initLevel[root] = 0;
        CUDA_CHECK(cudaMemcpy(level_d, initLevel.data(), numVertices * sizeof(int), cudaMemcpyHostToDevice));
        int currLevel = 1;
        int flag_h;
        do {
            CUDA_CHECK(cudaMemset(flag_d, 0, sizeof(int)));
            bfsPushKernel<<<gridDim, blockDim>>>(srcPtrs_d, dst_d, numVertices, level_d, currLevel, flag_d);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(&flag_h, flag_d, sizeof(int), cudaMemcpyDeviceToHost));
            ++currLevel;
        } while (flag_h);
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
    CUDA_CHECK(cudaFree(flag_d));
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
    float ms = runBfsPush(srcPtrs, dst, numVertices, root, level);

    bool ok = checkExact(level, ref);
    printf("%s (V=%d, E=%zu): %.4f ms  [%s]\n", name, numVertices, edges.size(), ms, ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    printf("BFS: vertex-centric push, one thread per vertex per level (§18.3, Fig. 18.6):\n");
    bool ok = true;
    ok = runCase("small graph (root=0)", 13, 0, smallGraphEdges()) && ok;
    ok = runCase("random 4000-vertex graph", 4000, 0, randomGraphEdges(4000, 6, 123u)) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
