# Chapter 18 — Graph Traversal

Section-by-section summary of the book content. Page numbers are printed
book pages (Chapter 18 runs pp. 425–451).

## 18.1 Background (p. 425)

A **graph** represents relations between entities: entities are
**vertices**, relations are **edges** (Fig. 18.1 — nine vertices,
fifteen directional edges). Graphs underlie many real-world problems
(social networks, driving-direction map services). The chapter focuses
on **directional** edges (a bidirectional relation is just two opposite
directional edges). An **adjacency matrix** `A` is the intuitive dense
representation: `A_{i,j} = 1` iff an edge goes from vertex `i` to
vertex `j` (Fig. 18.2); a fully-connected `N`-vertex graph would need
`N·(N-1)` such edges, but most real graphs are **sparsely connected**
(far fewer edges per vertex than `N-1`) — exactly the situation
Chapter 17's sparse matrix formats target.

Fig. 18.3 represents the example graph in **CSR**, **CSC**, and
**COO** — reusing Chapter 17's formats directly, but with graph-
specific naming: the row/column-offset arrays become `srcPtrs`/
`dstPtrs`, and the column-index array becomes `dst` (because a non-
zero's column index *is* the destination vertex of the represented
edge). Since the example graph's adjacency values are always 1 (an
edge either exists or doesn't), the `data` array itself is often
**implicit** and omitted entirely — only stored explicitly if edges
carry extra information (e.g. distance, connection date). Real graphs'
**degree distributions** (edges per vertex) vary by domain: road
networks have a relatively uniform, low average degree; social
networks have a much broader distribution (a few high-degree "hub"
vertices) — this structural difference matters for algorithm choice
throughout the chapter. Crucially, **different sparse formats give
different accessibility**: CSR gives easy access to a vertex's outgoing
edges, CSC to its incoming edges, COO to a given edge's source/
destination — so the choice of graph representation is tightly coupled
to the choice of traversal algorithm, a theme explored via **breadth-
first search (BFS)** for the rest of the chapter.

## 18.2 Breadth-first search (p. 429)

BFS discovers the shortest number of edges ("hops") from a **root**
vertex to every other reachable vertex, labeling each vertex with its
**level** (hop count) — Fig. 18.4 shows two complete worked examples
with different root vertices, illustrating that results differ
substantially even for roots only one edge apart. The set of labeled
vertices/traversed edges forms a **BFS tree**; the BFS tree lets one
reconstruct a shortest path from root to any target by **tracing back**
through predecessors whose level is exactly one less than the current
vertex (if multiple same-level predecessors exist, any is a valid
choice, meaning multiple equally-short paths exist). A worked real-
world application: **maze routing** in integrated-circuit CAD (Fig.
18.5) — the chip is modeled as a graph of wiring blocks, and BFS from
one net terminal discovers the shortest wire route to another,
respecting blockages from already-placed components/wires.

## 18.3 Vertex-centric parallelization of BFS (p. 431)

Two broad parallelization strategies recur throughout graph computing:
**vertex-centric** (assign threads to vertices, each operating on its
neighbors — via outgoing edges, incoming edges, or both) and
**edge-centric** (assign threads to edges, each looking up that edge's
source/destination — covered in §18.4). All of the chapter's
implementations share one structural choice: process BFS **one level
at a time**, launching a separate kernel per level, since all vertices
of a level must finish being labeled before the next level's labeling
begins (a cross-level barrier). (§18.7 later revisits this to fuse
levels into a single kernel.)

**Push (top-down) implementation** (Figs. 18.6–18.7): one thread per
vertex; a thread whose vertex is in the *previous* level iterates over
its vertex's **outgoing** edges (needs **CSR**), labeling any unvisited
neighbor as the current level and setting a "new vertex visited" flag
(used by host code to decide whether another level's kernel launch is
needed). `level` is initialized to a sentinel `UNVISITED` so a thread
can tell whether a neighbor has already been labeled. Multiple threads
may redundantly label the *same* neighbor — a technical race condition,
but benign/**idempotent** since every writer writes the identical
value (though this still violates the strict C++ memory model; a
conservative implementation would use an atomic here).

**Pull (bottom-up) implementation** (Figs. 18.8–18.9): one thread per
vertex; a thread whose vertex is **still unvisited** iterates over its
vertex's **incoming** edges (needs **CSC**), checking if any neighbor
belongs to the previous level — if so, labels its *own* vertex as the
current level and **breaks out of the loop early** (only one previous-
level neighbor is needed to justify the label; checking the rest is
wasted work). Comparing push vs. pull: (1) pull's early-break can save
substantial work versus push's full-neighbor-list scan, especially on
high-degree/high-variance graphs (social networks) — for low-degree/
low-variance graphs (road networks, CAD) the difference matters less;
(2) push only launches "useful" threads for previous-level vertices
(a small set, especially at early levels, when few vertices total are
unvisited) while pull launches threads for *every* still-unvisited
vertex (a large set early on) — making **push typically faster at
early levels**, and **pull typically faster at later levels** (once
most vertices are visited and early-break kicks in often). This
motivates a **direction-optimized** implementation that switches from
push to pull partway through — the right switch point depends on graph
structure, with **small-world graphs** (high average degree/variance,
few levels, rapid growth — e.g. social networks) switching much earlier
than **low-degree/variance** graphs (many levels, slow growth — e.g.
road networks). A direction-optimized implementation needs **both** CSR
and CSC stored — though for *undirected* graphs (symmetric adjacency
matrix, as in social networks or maze routing) CSR and CSC coincide, so
only one needs to be stored and both implementations can share it.

## 18.4 Edge-centric parallelization of BFS (p. 436)

One thread per **edge** (needs **COO**, Figs. 18.10–18.11): each thread
checks if its edge's source vertex is in the previous level, and if so,
labels the unvisited destination vertex as current-level. Two
advantages over vertex-centric: (1) **more exposed parallelism** — a
graph typically has far more edges than vertices, so edge-centric
launches more threads, which matters especially for **small graphs**
where a vertex-centric launch might not fill the device; (2) **less
load imbalance/divergence** — every thread handles exactly one edge
(uniform work), vs. vertex-centric threads whose work scales with
that vertex's (possibly very uneven) degree — an instance of Chapter 6's
rearranging-thread-to-data-mapping technique. Disadvantage: edge-centric
checks **every edge in the graph every level**, even edges whose source
vertex has long been determined irrelevant (vertex-centric can instead
skip a whole neighbor list with just one thread's one check); also, COO
needs more storage than CSR/CSC (an explicit `src` array per edge, where
CSR/CSC store structurally-grouped offsets instead).

A sidebar connects all these kernels to Chapter 17's SpMV code: a BFS
level iteration can be expressed almost entirely as SpMV plus a few
vector operations — the general **linear algebraic formulation** of
graph problems (standardized as the **GraphBLAS** API), whose advantage
is reusing mature, highly-optimized sparse linear algebra libraries, at
the cost of potentially missing algorithm-specific optimizations.

## 18.5 Improving work efficiency with frontiers (p. 438)

All implementations so far check **every** vertex or edge at **every**
level, launching many threads that simply discover irrelevance and do
no useful work — fully parallel and synchronization-free, but not
**work-efficient**. Quantified for a graph with `n` vertices, `m` edges,
diameter `d`: ideal work is `O(n+m)` (visit everything exactly once);
push achieves `O(d·n+m)` (every vertex checked every level, each edge
traversed once); pull achieves `O(d·n+d·m)` (every vertex *and* every
edge potentially re-checked across levels until found); edge-centric
achieves `O(d·m)` (every edge checked every level) — **none** of these
matches the ideal.

Fix: have threads processing the previous level **collaboratively
build a frontier** — an explicit list of exactly the vertices that need
checking next — so each level's kernel only launches threads for
*frontier* vertices, achieving the ideal `O(n+m)`. Fig. 18.12's push
kernel with frontiers: a thread per **frontier entry** (not per vertex)
looks up its actual vertex index from `prevFrontier[i]`, iterates its
outgoing edges, and for each successfully-labeled neighbor, **appends**
it to `currFrontier` via a device-scope atomic `fetch_add` on a frontier-
size counter (`numCurrFrontier`) to claim a unique insertion slot. The
host now detects "BFS complete" via an **empty** frontier, rather than a
flag.

Correctly adding a neighbor to the frontier **exactly once** (even
though multiple threads may simultaneously discover the same unvisited
neighbor) requires atomically **checking-and-labeling** the neighbor in
one indivisible step — the earlier "redundant labeling is harmless"
reasoning (§18.3) no longer applies, since redundant *frontier
insertion* would add the same vertex to the frontier multiple times,
wastefully reprocessing it next level. Fig. 18.14's
`visitVertexAtomically` device function uses C++'s
**compare-and-swap** (`cuda::atomic_ref::compare_exchange_strong`): it
atomically compares `level[vertex]` against `UNVISITED` and, only if
equal, sets it to `currLevel`, returning whether the swap succeeded —
giving each vertex exactly one "winning" thread, which is the only one
that inserts it into the frontier. `memory_order_relaxed` is used for
both the success and failure case, since no additional ordering
relative to other independent memory accesses is needed here. Dropping
redundant frontier insertions this way removes most of the wasted work,
at the cost of introducing genuine (moderate, for the compare-and-swap;
**high**, for the frontier-counter increment, since *every* successful
thread contends on the same single counter) atomic-operation
contention.

## 18.6 Reducing contention with privatization (p. 442)

The frontier-insertion pattern is structurally identical to Chapter
12's unstable filter — so its optimizations (**coalesced atomics**,
assumed compiler-applied, and **privatization**) transfer directly.
**Privatization**: each thread **block** maintains its own private
frontier and counter in **shared memory** (`currFrontier_s`,
`numCurrFrontier_s`), so threads mostly contend only with other threads
in the *same block*, not the whole grid (Figs. 18.15–18.16). A thread
that successfully visits a neighbor atomically increments the block's
private counter; if the private frontier isn't yet full, the neighbor
is inserted there (low-latency shared-memory atomic); if the private
frontier has **overflowed** its fixed capacity, the thread instead
falls back to a device-scope atomic directly on the public frontier (a
safety valve, used only when privatization's fixed-size buffer runs
out). After all threads in a block finish (a `__syncthreads()`), one
thread reserves a contiguous region in the **public** frontier (one
device-scope atomic *per block*, not per vertex) sized to the private
frontier's final count, and the block **collaboratively, coalesced-ly**
copies its private frontier into that public region — the same
privatize-then-commit pattern as Chapter 9 (histogram) and Chapter 12
(filter).

## 18.7 Reducing launch overhead with cooperative groups (p. 445)

Launching a separate kernel per BFS level is negligible overhead for
high-degree graphs (large frontiers, lots of work per level) but can
**dominate** total runtime for low-degree graphs (road networks — small
frontiers, little work per level, but potentially many levels/launches).
Fix: perform the **entire** BFS computation in a **single** kernel
launch, using **cooperative groups**' grid-wide barrier
(`grid.sync()`) between levels instead of kernel termination.

This requires a correctness guarantee cooperative-groups barriers don't
provide automatically: all participating thread blocks must be
**simultaneously resident** on the device, or a grid-wide barrier can
**deadlock** (scheduled blocks wait on not-yet-scheduled blocks, which
in turn wait on scheduled blocks to finish and vacate room for them).
The fix is to launch **no more blocks than can run simultaneously**,
computed via `cudaOccupancyMaxActiveBlocksPerMultiprocessor()` (blocks
per SM, Chapter 4's occupancy API) times the device's SM count
(`cudaDeviceProp.multiProcessorCount`) — then launch via the special
`cudaLaunchCooperativeKernel()` API (not ordinary `<<<...>>>` syntax),
passing kernel arguments packed into a `void*` array.

Fig. 18.17's complete multi-level kernel: an outer `for` loop over
levels runs **inside the kernel** as long as the frontier isn't empty;
its body closely mirrors Fig. 18.15's single-level, privatized,
frontier-based logic, with one key difference — since thread count is
now capped at the occupancy-safe maximum (possibly fewer than one
thread per frontier vertex), each thread loops over **multiple**
frontier entries using a grid-stride pattern (`grid.thread_rank()` /
`grid.num_threads()`, analogous to `blockIdx`/`blockDim`-based striding
but grid-wide). After each level, swapping `prevFrontier`/
`currFrontier` for the next iteration needs **three** separate
`grid.sync()` barriers: one ensuring all blocks finish adding to the
current frontier before its size is read; one ensuring all blocks
finish *reading* that size (into the new previous-frontier count)
before any thread resets the counter to 0; one ensuring the reset
completes before the next level starts adding to it again — mirroring
the double-buffering reasoning from Chapter 6, generalized to a
grid-wide (not just block-wide) scope; two of the three barriers could
be eliminated by using separate counters per level and resetting them
before the kernel starts, left as an exercise. Beyond eliminating
per-level launch overhead, fusing the whole computation into one kernel
also frees the host CPU to do other work concurrently, and — in
contexts beyond this specific example — can let data stay resident in
shared memory/registers across what would otherwise be separate kernel
launches (an advantage also exploited by Chapter 11's grid-wide
single-kernel scan, there via unidirectional rather than full barrier
synchronization).

## 18.8 Other optimizations (p. 448)

Addresses the **remaining vertex-centric load-imbalance** problem
directly (one or a few extremely-high-degree "celebrity" vertices in
graphs like social networks can make a handful of threads take vastly
longer than the rest, stalling the whole grid): edge-centric
parallelization (§18.4) is one fix already covered; another is
**sorting frontier vertices into degree-based buckets** (small, medium,
large) and processing each bucket with an appropriately-sized
parallelism granularity — one *thread* per vertex for the small bucket,
one *warp* per vertex for the medium bucket, one *thread block* per
vertex for the large bucket — a cited real implementation uses exactly
this three-bucket scheme, particularly effective on graphs with high
degree variance.

## 18.9 Summary (p. 449)

BFS served as the vehicle for the chapter's graph-traversal challenges:
representing graphs via sparse formats (reusing Chapter 17 directly,
with format choice tied to needed accessibility); the vertex-centric
vs. edge-centric design space and its tradeoffs; eliminating redundant
work via explicit **frontiers**; reducing atomic contention via
**privatization**; eliminating kernel-launch overhead via **cooperative
groups**' grid-wide synchronization; and briefly, bucket-based load
balancing for highly skewed-degree graphs. Though BFS is among the
simplest graph algorithms, it exhibits challenges characteristic of
much more complex graph computations generally: problem decomposition
for parallelism, privatization, fine-grained load balancing, and
correct synchronization. NVIDIA's **RAPIDS cuGraph** library already
provides efficient parallel implementations of many graph algorithms in
practice; this chapter's techniques equip a reader to use such libraries
well, and to build new graph algorithms when a library doesn't already
cover the need — including handling graphs too large for GPU memory, or
preprocessing/reordering a graph's vertices to expose more parallelism,
locality, or load balance.

## 18.10 Exercises (p. 449)

Three problems: a multi-part exercise on a given 8-vertex directed
graph — represent it as an adjacency matrix (a) and in CSR with sorted
neighbor lists (b), then for a BFS from vertex 0, compute exactly how
many threads are launched and how many iterate over neighbors (and, for
pull, how many actually label their vertex) under each of push, pull,
edge-centric, and frontier-based push implementations (c); implement
the host code for the direction-optimized BFS implementation described
in §18.3 (Q2); and modify the cooperative-groups kernel (Fig. 18.17) to
use only a single grid-wide barrier per level instead of three (Q3,
i.e. §18.7's deferred optimization). Not implemented in this repo's
samples (end-of-chapter exercises are out of scope — see the repo root
README) — though the repo's own `06_bfs_vertex_centric_pull.cu` sample
implements §18.3's pull (bottom-up) kernel head-to-head against the
existing push kernel, which the book presents as a full worked
implementation (Figs. 18.8–18.9) rather than a numbered exercise.
