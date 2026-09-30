// Chapter 18: Graph traversal
// §18.3, Fig. 18.8: Vertex-centric pull (bottom-up) BFS kernel, head-to-head
// against the push (top-down) kernel from file 01 (Fig. 18.6).
//
// §18.3 gives TWO complete kernel listings for vertex-centric BFS: the push
// kernel (Fig. 18.6, file 01) and this file's pull kernel (Fig. 18.8) --
// but only the push variant was previously implemented anywhere in this
// chapter, so the book's own push-vs-pull comparison was never actually
// measured. This file implements the pull kernel and times it directly
// against a local copy of the push kernel on the identical test graphs
// file 01 uses (same edge lists, same seed), so the two are directly
// comparable.
//
// Pull, mechanically: one thread per vertex, every level. Each thread whose
// vertex is still UNVISITED walks its vertex's INCOMING edges (via a CSC
// dstPtrs/src representation -- the opposite accessibility from push's CSR)
// looking for a neighbor already labeled at the previous level; the first
// one found labels this vertex and BREAKS out of the loop immediately
// (Fig. 18.8, line 12) -- unlike push, which always walks every one of its
// vertex's edges. §18.3's own tradeoff analysis, reproduced in the README:
// push only launches real work for previous-level vertices and always
// finishes its edge list; pull launches a thread for every still-unvisited
// vertex regardless of level and may exit its edge list early. Neither
// dominates unconditionally -- early levels (few previous-level vertices,
// many unvisited) favor push; later levels (many previous-level vertices,
// most vertices already visited, an unvisited vertex's neighbors likelier
// to already be visited -> more early exits) favor pull. This file measures
// whole-traversal wall-clock time for each strategy run in isolation (not
// the direction-optimized per-level switching hybrid §18.3 mentions in
// closing, which needs a switching heuristic no figure in this chapter
// gives -- out of scope, consistent with this project's convention).

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <queue>
#include <vector>

#include "../../common/cuda_utils.h"

const int UNVISITED = INT_MAX;

// ---------------------------------------------------------------------------
// §18.3, Fig. 18.6: vertex-centric push BFS kernel (identical to file 01,
// duplicated here for a direct, same-binary timing comparison).
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

// ---------------------------------------------------------------------------
// §18.3, Fig. 18.8: vertex-centric pull BFS kernel. One thread per vertex;
// only still-unvisited vertices do any work, and a thread breaks out of its
// incoming-edge loop the instant it finds a previous-level neighbor.
// ---------------------------------------------------------------------------
__global__ void bfsPullKernel(const int *dstPtrs, const int *src, int numVertices, int *level, int currLevel, int *newVertexVisited) {
    int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex < numVertices) {
        if (level[vertex] == UNVISITED) {
            for (int edge = dstPtrs[vertex]; edge < dstPtrs[vertex + 1]; ++edge) {
                int neighbor = src[edge];
                if (level[neighbor] == currLevel - 1) {
                    level[vertex] = currLevel;
                    *newVertexVisited = 1;
                    break;
                }
            }
        }
    }
}

struct Edge {
    int src, dst;
};

// Identical to file 01's small graph: 13 vertices, root 0, vertex 12
// deliberately left unreachable.
std::vector<Edge> smallGraphEdges() {
    return {
        {0, 1}, {0, 2}, {1, 3}, {1, 4}, {2, 4}, {2, 5}, {3, 6}, {4, 6}, {4, 7}, {5, 7}, {5, 8}, {6, 9}, {7, 9}, {7, 10}, {8, 10}, {9, 11}, {10, 11}};
}

// Identical to file 01's random-graph generator (same seed -> same edges).
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

// CSC: groups the same edges by DESTINATION, giving each vertex's incoming
// edges -- the accessibility the pull kernel needs (§18.3).
void buildCsc(const std::vector<Edge> &edges, int numVertices, std::vector<int> &dstPtrs, std::vector<int> &src) {
    dstPtrs.assign(numVertices + 1, 0);
    for (const auto &e : edges) dstPtrs[e.dst + 1]++;
    for (int v = 0; v < numVertices; ++v) dstPtrs[v + 1] += dstPtrs[v];
    src.assign(edges.size(), 0);
    std::vector<int> cursor(dstPtrs.begin(), dstPtrs.end() - 1);
    for (const auto &e : edges) src[cursor[e.dst]++] = e.src;
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

// Runs one BFS strategy's kernel to convergence, with its own untimed
// warm-up run immediately before its own timed run (this project's
// timing-fairness rule for files that time two kernels in one process).
template <typename KernelFn>
float runBfs(KernelFn kernel, const int *ptrs_d, const int *idx_d, int numVertices, int root, int *level_d, int *flag_d) {
    auto runOnce = [&]() {
        std::vector<int> initLevel(numVertices, UNVISITED);
        initLevel[root] = 0;
        CUDA_CHECK(cudaMemcpy(level_d, initLevel.data(), numVertices * sizeof(int), cudaMemcpyHostToDevice));
        int blockDim = 256;
        int gridDim = (numVertices + blockDim - 1) / blockDim;
        int currLevel = 1;
        int flag_h;
        do {
            CUDA_CHECK(cudaMemset(flag_d, 0, sizeof(int)));
            kernel<<<gridDim, blockDim>>>(ptrs_d, idx_d, numVertices, level_d, currLevel, flag_d);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(&flag_h, flag_d, sizeof(int), cudaMemcpyDeviceToHost));
            ++currLevel;
        } while (flag_h);
    };

    runOnce();  // warm-up
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    runOnce();
    return timer.stopAndGetMs();
}

bool checkExact(const std::vector<int> &a, const std::vector<int> &b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) return false;
    return true;
}

bool runCase(const char *name, int numVertices, int root, const std::vector<Edge> &edges) {
    std::vector<int> srcPtrs, dst, dstPtrs, src;
    buildCsr(edges, numVertices, srcPtrs, dst);
    buildCsc(edges, numVertices, dstPtrs, src);

    std::vector<int> ref = cpuBfs(srcPtrs, dst, numVertices, root);

    int numEdges = static_cast<int>(edges.size());
    int *srcPtrs_d, *dst_d, *dstPtrs_d, *src_d, *level_d, *flag_d;
    CUDA_CHECK(cudaMalloc(&srcPtrs_d, (numVertices + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dst_d, numEdges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dstPtrs_d, (numVertices + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&src_d, numEdges * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&level_d, numVertices * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&flag_d, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(srcPtrs_d, srcPtrs.data(), (numVertices + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dst_d, dst.data(), numEdges * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dstPtrs_d, dstPtrs.data(), (numVertices + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(src_d, src.data(), numEdges * sizeof(int), cudaMemcpyHostToDevice));

    float pushMs = runBfs(bfsPushKernel, srcPtrs_d, dst_d, numVertices, root, level_d, flag_d);
    std::vector<int> pushLevel(numVertices);
    CUDA_CHECK(cudaMemcpy(pushLevel.data(), level_d, numVertices * sizeof(int), cudaMemcpyDeviceToHost));

    float pullMs = runBfs(bfsPullKernel, dstPtrs_d, src_d, numVertices, root, level_d, flag_d);
    std::vector<int> pullLevel(numVertices);
    CUDA_CHECK(cudaMemcpy(pullLevel.data(), level_d, numVertices * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(srcPtrs_d));
    CUDA_CHECK(cudaFree(dst_d));
    CUDA_CHECK(cudaFree(dstPtrs_d));
    CUDA_CHECK(cudaFree(src_d));
    CUDA_CHECK(cudaFree(level_d));
    CUDA_CHECK(cudaFree(flag_d));

    bool pushOk = checkExact(pushLevel, ref);
    bool pullOk = checkExact(pullLevel, ref);
    printf("%s (V=%d, E=%zu): push %.4f ms [%s]  |  pull %.4f ms [%s]\n", name, numVertices, edges.size(),
           pushMs, pushOk ? "match" : "MISMATCH", pullMs, pullOk ? "match" : "MISMATCH");
    return pushOk && pullOk;
}

int main() {
    printf("BFS: vertex-centric push (Fig. 18.6) vs pull (Fig. 18.8), head-to-head (§18.3):\n");
    bool ok = true;
    ok = runCase("small graph (root=0)", 13, 0, smallGraphEdges()) && ok;
    ok = runCase("random 4000-vertex graph", 4000, 0, randomGraphEdges(4000, 6, 123u)) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
