// Chapter 18: Graph traversal
// §18.4, Fig. 18.10: Edge-centric BFS kernel.
//
// One thread per EDGE. Each level, every thread looks up the source and
// destination vertex of its own edge (via the COO src/dst arrays) and
// checks whether the source belongs to the previous level and the
// destination is unvisited; if so, it labels the destination as belonging
// to the current level. Since a graph typically has many more edges than
// vertices, this launches more threads than the vertex-centric kernels
// (01_bfs_vertex_centric.cu) and is therefore more suitable for small
// graphs where a vertex-centric launch might not fully occupy the device.
// It also exhibits less load imbalance/control divergence, since every
// thread does the same fixed amount of work (inspect one edge) instead of
// looping over a variable-length neighbor list. The tradeoffs: every edge
// is checked at every level (O(d*m) work, §18.5) and COO needs more
// storage than CSR/CSC.

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <queue>
#include <vector>

#include "../../common/cuda_utils.h"

const int UNVISITED = INT_MAX;

// ---------------------------------------------------------------------------
// §18.4, Fig. 18.10: edge-centric BFS kernel -- one thread per edge, called
// once per level.
// ---------------------------------------------------------------------------
__global__ void bfsEdgeCentricKernel(const int *src, const int *dst, int numEdges, int *level, int currLevel, int *newVertexVisited) {
    int edge = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge < numEdges) {
        int vertex = src[edge];
        if (level[vertex] == currLevel - 1) {
            int neighbor = dst[edge];
            if (level[neighbor] == UNVISITED) {
                level[neighbor] = currLevel;
                *newVertexVisited = 1;
            }
        }
    }
}

struct Edge {
    int src, dst;
};

// Same small graph as 01_bfs_vertex_centric.cu (13 vertices, root 0, vertex
// 12 unreachable) so results are directly comparable across the chapter's
// files.
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

float runBfsEdgeCentric(const std::vector<Edge> &edges, int numVertices, int root, std::vector<int> &level_h) {
    int numEdges = static_cast<int>(edges.size());
    std::vector<int> src_h(numEdges), dst_h(numEdges);
    for (int i = 0; i < numEdges; ++i) {
        src_h[i] = edges[i].src;
        dst_h[i] = edges[i].dst;
    }

    int *src_d, *dst_d, *level_d, *flag_d;
    CUDA_CHECK(cudaMalloc(&src_d, numEdges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dst_d, numEdges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&level_d, numVertices * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&flag_d, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(src_d, src_h.data(), numEdges * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dst_d, dst_h.data(), numEdges * sizeof(int), cudaMemcpyHostToDevice));

    int blockDim = 256;
    int gridDim = (numEdges + blockDim - 1) / blockDim;

    auto runOnce = [&]() {
        std::vector<int> initLevel(numVertices, UNVISITED);
        initLevel[root] = 0;
        CUDA_CHECK(cudaMemcpy(level_d, initLevel.data(), numVertices * sizeof(int), cudaMemcpyHostToDevice));
        int currLevel = 1;
        int flag_h;
        do {
            CUDA_CHECK(cudaMemset(flag_d, 0, sizeof(int)));
            bfsEdgeCentricKernel<<<gridDim, blockDim>>>(src_d, dst_d, numEdges, level_d, currLevel, flag_d);
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

    CUDA_CHECK(cudaFree(src_d));
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
    float ms = runBfsEdgeCentric(edges, numVertices, root, level);

    bool ok = checkExact(level, ref);
    printf("%s (V=%d, E=%zu): %.4f ms  [%s]\n", name, numVertices, edges.size(), ms, ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    printf("BFS: edge-centric, one thread per edge per level (§18.4, Fig. 18.10):\n");
    bool ok = true;
    ok = runCase("small graph (root=0)", 13, 0, smallGraphEdges()) && ok;
    ok = runCase("random 4000-vertex graph", 4000, 0, randomGraphEdges(4000, 6, 123u)) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
