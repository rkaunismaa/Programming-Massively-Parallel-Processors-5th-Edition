# Chapter 18: Graph traversal

Source: *Programming Massively Parallel Processors*, 5th ed., Ch. 18 (pp. 425-451).

| File | Book section | Technique |
|------|-------------|-----------|
| `01_bfs_vertex_centric.cu` | §18.3 | Vertex-centric push (top-down) BFS: one thread per vertex per level, unsynchronized (benign-race) neighbor labeling (Fig. 18.6) |
| `02_bfs_edge_centric.cu` | §18.4 | Edge-centric BFS: one thread per edge per level, using a COO representation (Fig. 18.10) |
| `03_bfs_frontier_based.cu` | §18.5 | Work-efficient vertex-centric push BFS with explicit frontiers and atomic compare-and-swap labeling (Figs. 18.12, 18.14) |
| `04_bfs_frontier_privatized.cu` | §18.6 | Frontier-based BFS with block-local (shared-memory) private frontiers to reduce atomic contention (Fig. 18.15) |
| `05_bfs_cooperative_groups.cu` | §18.7 | Single-launch, multi-level BFS using cooperative-groups grid-wide barrier sync (Fig. 18.17) |

All five files implement breadth-first search (BFS): each builds a small
synthetic directed graph in CSR (or, for file 02, COO) form, computes a CPU
BFS reference (`level[v]` = fewest hops from a fixed root, `INT_MAX` for
unreachable vertices), runs the CUDA kernel(s), and compares the resulting
level array against the CPU reference with **exact** integer equality (per
Chapter 18's `level`/distance arrays being integer-valued, not floats), then
prints a timing line and `PASS`/`FAIL`.

**Test graphs.** §18.1-18.2's worked example (Figs. 18.1/18.4, a nine-vertex
graph with fifteen directed edges) is only shown as an image in the book;
`pdftotext -layout` extracts none of its edges, so it cannot be
reconstructed from the extracted text (the closest attempt from the prose
alone -- e.g. "there is one edge going from vertex 0 to vertex 1... vertex 3
(through vertex 1)... vertex 8 (through any of vertices 3, 4 or 6)" -- yields
only 13 of its 15 edges, and even that requires cross-referencing two
different root vertices' BFS results from Fig. 18.4(a) and 18.4(b)). Per the
task brief, every file therefore builds its own small synthetic test graph
instead: a fixed 13-vertex graph (root 0, five BFS levels, with vertex 12
deliberately left unreachable to exercise the `UNVISITED` sentinel), plus a
randomly generated 4000-vertex/~25,600-edge graph for a meaningful timing
comparison across the five parallelization strategies. Both graphs are
identical across all five files (down to the same edge list and the same
LCG-seeded random generator), so the timings in the Results section below
are directly comparable.

## §18.3 Vertex-centric push -- `01_bfs_vertex_centric.cu`

One thread per vertex, relaunched once per BFS level from the host. Each
thread checks whether its vertex belongs to the previous level and, if so,
walks the vertex's outgoing edges (CSR `srcPtrs`/`dst`) and labels every
unvisited neighbor as belonging to the current level, setting a
`newVertexVisited` flag the host reads back to decide whether to launch
another level. As §18.3 notes, multiple threads can race to label the same
neighbor or set the flag; since every racing write is identical
(idempotent), this is a **benign** race and no atomics are used, matching
Fig. 18.6 exactly. The host loop labels level 0 (the root), then repeatedly
launches the kernel for levels 1, 2, ... until a launch sets no new
vertices. Work is O(d*n + m) (§18.5): every vertex is re-checked at every
level.

*(§18.3 also covers a vertex-centric* pull *(bottom-up) kernel over
incoming/CSC edges, Fig. 18.8 -- the task brief scopes this chapter's file
01 to the* push *variant specifically, so the pull kernel is not
implemented here.)*

## §18.4 Edge-centric -- `02_bfs_edge_centric.cu`

One thread per edge, relaunched once per level. Each thread looks up its
edge's source and destination via COO `src`/`dst` arrays, checks if the
source belongs to the previous level, and if so labels an unvisited
destination as belonging to the current level (Fig. 18.10). Since a graph
typically has far more edges than vertices, this launches more threads than
the vertex-centric kernel and can better occupy the device on small graphs;
every thread also does a fixed, uniform amount of work (inspect one edge),
avoiding the load imbalance a vertex-centric thread suffers when its
vertex's neighbor-list length varies. The cost (§18.5): every edge is
checked at every level regardless of relevance, giving O(d*m) work, and COO
storage is larger than CSR/CSC.

## §18.5 Frontier-based -- `03_bfs_frontier_based.cu`

Files 01 and 02 both re-scan the *entire* vertex or edge set at every level.
This file instead launches one thread only for each element of an explicit
*frontier* array holding the previous level's vertices (Fig. 18.12), cutting
total work to the ideal O(n + m). Because several threads can now race to
discover the *same* unvisited neighbor and each would otherwise push a
duplicate entry into the next frontier, the "check unvisited, then label"
step must be a single atomic operation: `visitVertexAtomically` (Fig. 18.14)
uses `cuda::atomic_ref<int>::compare_exchange_strong` against the
`UNVISITED` sentinel to guarantee exactly one thread observes success per
vertex. A successful thread then atomically increments a global
`numCurrFrontier` counter (`atomicAdd`) and inserts itself into the current
frontier at the returned index. The host swaps the previous/current
frontier device buffers and passes the current frontier's size as next
level's launch size, stopping when the frontier is empty.

## §18.6 Privatized frontiers -- `04_bfs_frontier_privatized.cu`

File 03's `atomicAdd` on `numCurrFrontier` is contended by every thread in
the entire grid. This file applies the same privatization pattern as
Chapter 12's unstable filter: each thread BLOCK first accumulates its own
discoveries into a small shared-memory frontier (`currFrontier_s`,
capacity 32) with a block-local shared counter, so contention is limited to
threads within one block on low-latency shared memory (Fig. 18.15). If a
block's private frontier overflows its capacity, the overflowing thread
restores the shared counter to the capacity and falls back to inserting
directly into the public frontier via the global atomic. After all threads
in the block finish, one thread reserves a contiguous range in the public
frontier with a single `atomicAdd(numCurrFrontier, numCurrFrontier_s)`, and
the whole block copies its private frontier into that range with `threadIdx.x`-
indexed, and therefore coalesced, global writes.

## §18.7 Cooperative groups -- `05_bfs_cooperative_groups.cu`

Files 01-04 all relaunch a grid once per BFS level, because a level's kernel
must fully complete before the next level can safely start. When frontiers
are small, that relaunch overhead can dominate runtime. This file performs
the *entire* multi-level traversal in a **single** kernel launch, using the
cooperative-groups API's `grid.sync()` as a grid-wide barrier between
levels instead of a host-side relaunch (Fig. 18.17). A grid-wide barrier is
only safe if every block in the grid can be simultaneously resident on an
SM (otherwise unscheduled blocks waiting to be scheduled could deadlock
against scheduled blocks waiting at the barrier for them to finish), so the
host first computes the maximum concurrently resident block count via
`cudaOccupancyMaxActiveBlocksPerMultiprocessor(...) * multiProcessorCount`
and launches exactly that many blocks -- no more -- through
`cudaLaunchCooperativeKernel()`, which requires packing kernel arguments
into a `void*[]` array rather than using `<<<...>>>` syntax. Because the
grid can now have fewer threads than there are frontier vertices, each
thread processes a grid-stride loop of frontier elements
(`grid.thread_rank()`/`grid.num_threads()`) instead of the strict
one-thread-per-element mapping files 03/04 use. Per §18.7's kernel
argument list -- `numPrevFrontier` (no `_d` suffix) alongside
`numCurrFrontier_d` (a device pointer) -- `numPrevFrontier` is passed by
value and kept as a per-thread register updated after each `grid.sync()`
from a read of `*numCurrFrontier`, while `numCurrFrontier` is the one
counter that actually lives in device memory (all threads in the grid
atomically add to it). Before use, the program calls
`cudaDeviceGetAttribute(cudaDevAttrCooperativeLaunch, ...)` and prints a
clear skip message (exiting 0, not a `FAIL`) if the device doesn't support
cooperative grid launch; both GPUs this repo was built and tested on
(RTX 2070 SUPER, RTX 4090) do support it, so the skip path is defensive
code, not what actually runs here.

**A build note on `-G` and cooperative launch.** This chapter's Makefile is
debug-by-default (`-O0 -g -G`) like the rest of the repo, but device debug
info (`-G`) specifically causes `05_bfs_cooperative_groups.cu`'s
`grid.sync()` to hang inside `cudaDeviceSynchronize()` on this repo's
hardware: `cudaDeviceGetAttribute` still reports cooperative launch as
supported, but the forced device-debug code generation appears to break the
simultaneous-SM-residency guarantee a grid-wide barrier depends on.
Isolated by binary search over flags (`-O0 -g` alone: runs correctly;
adding `-G`: reproduces the hang; `-O2`: runs correctly), so the `Makefile`
builds this one file without `-G` even when `DEBUG=1` (still `-O0 -g` for
host-side debuggability), leaving the other four files unaffected.

## Results

```
== bin/01_bfs_vertex_centric ==
BFS: vertex-centric push, one thread per vertex per level (§18.3, Fig. 18.6):
small graph (root=0) (V=13, E=17): 0.1099 ms  [match]
random 4000-vertex graph (V=4000, E=25592): 0.1758 ms  [match]
PASS
== bin/02_bfs_edge_centric ==
BFS: edge-centric, one thread per edge per level (§18.4, Fig. 18.10):
small graph (root=0) (V=13, E=17): 0.0895 ms  [match]
random 4000-vertex graph (V=4000, E=25592): 0.1390 ms  [match]
PASS
== bin/03_bfs_frontier_based ==
BFS: vertex-centric push with frontiers, work-efficient O(n+m) (§18.5, Fig. 18.12/18.14):
small graph (root=0) (V=13, E=17): 0.1515 ms  [match]
random 4000-vertex graph (V=4000, E=25592): 0.8172 ms  [match]
PASS
== bin/04_bfs_frontier_privatized ==
BFS: vertex-centric push with privatized (block-local) frontiers (§18.6, Fig. 18.15):
small graph (root=0) (V=13, E=17): 0.1519 ms  [match]
random 4000-vertex graph (V=4000, E=25592): 0.9580 ms  [match]
PASS
== bin/05_bfs_cooperative_groups ==
BFS: single-launch multi-level kernel with cooperative-groups grid sync (§18.7, Fig. 18.17):
small graph (root=0) (V=13, E=17): 0.0543 ms  [match]
random 4000-vertex graph (V=4000, E=25592): 0.1478 ms  [match]
PASS
```

All five binaries also pass under `compute-sanitizer --tool memcheck` (0
errors) at `-arch=sm_75` -- notable here since files 03, 04, and 05 rely on
`cuda::atomic_ref::compare_exchange_strong` and `atomicAdd` for frontier
construction, and file 05 additionally performs a grid-wide cooperative
launch.

The random-graph timings above are not a fair speed comparison of the five
strategies in the way the numbers might suggest: the frontier-based kernels
(03, 04, 05) pay per-level `cudaMemcpy`/kernel-launch overhead relative to
their small per-level frontiers on this graph, while 01/02's non-frontier
approach happens to win on raw wall-clock here specifically because the
test graph is small enough that re-scanning every vertex/edge each level is
still cheap -- the O(n+m) vs O(d*n+m)/O(d*m) work-efficiency argument in
§18.5 is about asymptotic behavior on much larger graphs, not a guarantee
for a 4000-vertex synthetic graph run once.

Build and run all samples in this chapter:

```sh
make run
```
