# Chapter 16 — Dynamic Programming and Wavefront Parallelism

Section-by-section summary of the book content. Page numbers are printed
book pages (Chapter 16 runs pp. 373–400).

## 16.1 Dynamic programming (p. 373)

**Dynamic programming** solves problems with **optimal substructure**
and **overlapping sub-problems**: the original problem's solution can
be recursively assembled from smaller sub-problems' solutions, and
those sub-problems recur often enough that caching their solutions
(rather than recomputing them) pays off. This distinguishes dynamic
programming from **divide-and-conquer** (e.g. merge sort, Chapter 14),
whose sub-problems don't overlap and so need no such caching. The
classic illustrative example is the Fibonacci recurrence
`F_i = F_{i-1} + F_{i-2}` (Eq. 16.1) — naive recursion re-solves the
same `F_{i-2}` repeatedly; dynamic programming avoids this by storing
sub-problem solutions in a table, via either **memoization** (cache
results as they're computed, top-down) or **tabulation** (compute
results bottom-up into a table), covered in §16.2. The set of
sub-problems solved together in one step of the computation is a
**wavefront** — since wavefront members don't depend on each other,
wavefronts are a natural unit of GPU parallelism. Other dynamic
programming applications named: shortest-path algorithms (Dijkstra,
Bellman-Ford, **Floyd-Warshall** — this chapter's §16.4 example), chain
matrix multiplication, time series analysis (dynamic time warping),
Hidden Markov models (Viterbi), and bioinformatics (**Smith-Waterman**
and Needleman-Wunsch sequence alignment — this chapter's §16.5 example
— plus protein folding, RNA structure prediction, protein-DNA
binding).

## 16.2 Implementation approaches (p. 374)

Two implementation styles, illustrated on Fibonacci (Fig. 16.1):
**top-down** uses recursion plus memoization — each call first checks
an associative array (`hash_table`) for an already-computed result,
computing and caching it only if missing. **Bottom-up** is iterative —
start from the smallest sub-problems and tabulate progressively larger
ones in a simple array (`table`), guaranteeing every sub-problem a
larger computation needs is already solved by the time it's needed.

Top-down is conceptually simple and matches the mathematical recurrence
directly, but is a **poor fit for GPUs** for two reasons: (1) CUDA
kernels don't efficiently support deep nested/recursive function calls
(device functions are normally inlined; true recursion can't be, which
hurts performance significantly); (2) accesses to an associative array
like a hash table tend to be scattered/uncoalesced across threads in a
warp. The chapter therefore focuses entirely on **bottom-up**
implementations for the rest of the chapter.

## 16.3 Wavefront patterns (p. 376)

A wavefront is the set of sub-problems solvable in parallel at one
step; successive wavefronts ("waves") are serialized by synchronizing
the parallel compute units (e.g. barriers) between them. Fig. 16.2
shows four example wavefront patterns (checkerboard, financial,
Smith-Waterman, H.264), differing in **dependence pattern** (which
cells feed which) and in whether **wavefront size is constant** across
iterations (checkerboard, financial — and this chapter's Floyd-Warshall
example, §16.4) or **changes** across iterations (Smith-Waterman,
H.264 — and this chapter's §16.5 example). A further distinguishing
factor: whether the **entire** table of sub-problem solutions must be
retained (Smith-Waterman needs the full table for a later traceback
step, §16.6) or whether older wavefronts' values can be **discarded**
once no longer needed (Fibonacci needs only the last two values;
Floyd-Warshall needs only one 2D plane of its conceptual 3D sub-problem
space at a time, discarding older planes).

## 16.4 Floyd-Warshall algorithm (p. 377)

Finds shortest paths between **all pairs** of vertices in a directed
weighted graph. Defining `d(i,j,k)` as the shortest `i→j` distance
using only intermediate vertices `0..k`, the recurrence
`d(i,j,k) = min(d(i,j,k-1), d(i,k,k-1) + d(k,j,k-1))` (either don't
route through `k`, or do) decomposes the problem into three smaller
sub-problems. Iterating `k` from `-1` (no intermediate vertices — just
the graph's direct edge weights, or infinity) up through `N-1` (all
vertices available) eventually yields every pair's true shortest
distance. Although the full sub-problem space is conceptually `O(N³)`,
only the current `k`-plane needs to be retained at once — `O(N²)` space
— since once `d(·,·,k)` is computed, `d(·,·,k-1)` can be discarded.
Pseudocode (Fig. 16.3): three nested loops (`k` outer, `i`/`j` inner);
all `(i,j)` cells within one `k` iteration form a wavefront computable
fully in parallel; iterations across `k` are serialized by a barrier
("Synchronize").

GPU kernel (Fig. 16.4): one thread per `(i,j)` pair, one kernel launch
per `k` iteration (host loop, Fig. 16.4's bottom snippet), with a
grid-wide kernel-termination barrier between iterations providing the
required cross-wavefront synchronization. Each thread block covers one
row of the distance table (2D grid: `blockIdx.y` = row, one-dimensional
blocks along columns); each thread reads its own cell plus the cell in
its row at column `k` (loaded once per block into shared memory,
`dist_k_col`) and the cell in column `col` at row `k`
(`dist_k_row`), updating its own cell if routing through `k` is
shorter. An important correctness observation: because `dist[k][k] = 0`
for non-negative-weight graphs, row-`k`/column-`k` cells are
mathematically guaranteed **unchanged** by the update formula applied
to themselves — so there's no read/write hazard requiring double
buffering (Chapter 6), and a single shared `dist` table suffices safely.
Floyd-Warshall parallelizes simply because of its regularity: constant
wavefront size, and dependencies limited to just row `k` and column `k`
from the previous iteration.

## 16.5 Genome sequence alignment and Smith-Waterman algorithm (p. 382)

**Genome sequence alignment** compares randomized DNA fragment "reads"
(from a sequencing machine) against a reference genome to detect
homology (shared ancestry). **Smith-Waterman** (like Needleman-Wunsch)
solves this via a dynamic programming **scoring matrix** `H`
(dimensions `(L_A+1)×(L_B+1)` for sequences `A`, `B`): row 0/column 0
are boundary zeros; cell `H_{i,j}` scores aligning segments
`A[1..j]`/`B[1..i]`, derived from three neighboring cells plus a direct
**similarity score** `S_{i,j}` (positive for a base-pair match, e.g.
+3; negative for a mismatch, e.g. −3) and **gap penalties**
(`deletion_penalty`/`insertion_penalty`, e.g. 2, for a missing/extra
base pair): `H_{i,j} = max(H_{i-1,j-1}+S_{i,j}, H_{i-1,j}−deletion_penalty,
H_{i,j-1}−insertion_penalty, 0)` (Eq. 16.2) — the floor at 0 keeps
scores non-negative. Fig. 16.6 depicts the resulting dependency
structure: each cell depends only on three cells in the **previous two
anti-diagonals**, so each **anti-diagonal of the matrix is itself a
wavefront** (the first anti-diagonal, wavefront 1, has only one cell;
subsequent wavefronts first grow, then shrink, as they cross the
matrix). Unlike Floyd-Warshall, the entire scoring matrix must be
retained to the end, since the subsequent **traceback** step (starting
from the cell with the highest score, following predecessor cells back
to a 0, to reconstruct the actual optimal alignment) needs it — but
traceback is inherently sequential, so the chapter's parallelization
effort (§16.6–16.8) focuses entirely on computing the wavefronts, not
on traceback.

## 16.6 Wavefront parallelization: block-level tiling (p. 384)

A **basic** GPU parallelization launches one kernel per anti-diagonal,
one thread per cell (Fig. 16.7a) — correct (kernel termination
guarantees a wavefront fully completes before the next starts) but
wasteful: no data reuse across adjacent anti-diagonals, and a
kernel-launch cost paid per anti-diagonal. (Left as an exercise.)

**Block-level tiling** (Fig. 16.7b) instead partitions the scoring
matrix into square tiles, one tile per thread block; within a tile,
threads compute successive intra-tile anti-diagonals using only a
local `__syncthreads()` — so a **global** synchronization (kernel
termination) is needed only once per "anti-diagonal of tiles," not per
individual anti-diagonal, and a tile's cells can be staged in **shared
memory** for faster, coalesced final writeback. Fig. 16.8's host code:
assuming equal-length sequences (`L = L_seq+1` square matrix), computes
`numTiles_x` tiles per dimension and loops over `2*numTiles_x - 1` tile
anti-diagonals, launching the kernel once per tile-anti-diagonal (with
the maximal possible block count each time, even though not all blocks
are active in every call — exact sizing left as an exercise).

Fig. 16.9's kernel: each thread block computes one `tile_width ×
tile_width` tile. Each thread determines its tile's row/column
(`blockIdx.x` = tile row; tile column computed from the tile-
anti-diagonal number `d` minus `blockIdx.x`, inactive if out of range)
and then loops over `2*tile_width - 1` intra-tile anti-diagonals,
computing one cell per active thread per iteration (bounds- and
tile-position-checked), reading its two anti-diagonal-back neighbors
and one two-back-diagonal neighbor via three small device functions
(`load_n`/`load_w`/`load_nw`, Fig. 16.10) that transparently fetch from
shared memory for interior tile cells or from global memory for
boundary cells at a tile's top/left edge (previously computed by a
neighboring tile/kernel call). `max4()` (Fig. 16.11) computes the
4-value max (3 candidates plus the 0 floor). After all intra-tile
anti-diagonals, `store_tile()` (Fig. 16.12) writes the whole tile back
to global memory row-by-row, **collaboratively and coalesced** — an
instance of the corner-turning/packing idea (Chapters 6, 12): turning
an inherently irregular access pattern into a coalesced one by staging
through shared memory first.

## 16.7 Hyperplane transformation (p. 389)

Square (or rectangular) tiling has two drawbacks (Fig. 16.13): (1)
**non-uniform intra-tile anti-diagonal lengths** mean the number of
active threads varies every iteration (some threads idle, warps
under-utilized), and a `n×n` tile needs `2n-1` iterations even though
peak parallelism (`n` active threads) is reached only briefly; (2)
**lost cross-tile locality** — only a tile's *last* anti-diagonals are
likely still in cache (e.g. L2) by the time an adjacent tile needs
them, since the *earlier* anti-diagonals a neighbor also needs were
computed (and likely evicted) much earlier.

The fix is **hyperplane transformation** (a.k.a. hyperplane
partitioning): an affine **shear transformation** turns rectangular
tiles into parallelogram **hypertiles**, by shifting each row of the
tile horizontally (by an amount proportional to its row index times a
**shear factor** `m`) so that what were *columns* in the original tile
become **anti-diagonals** in the transformed hypertile (Fig. 16.14).
This gives hypertiling three advantages over square tiling: (1)
**uniform** intra-tile anti-diagonal length (= the hypertile's own
width) every iteration, since all threads are now always active —
eliminating the idle-thread waste; (2) **fewer iterations** per tile
(`n` instead of `2n-1`, since more useful work gets done per iteration
— Eq. 16.3 vs. 16.4 show total work is identical, but hypertiling's
work-per-iteration average is `n` vs. square tiling's `n/2`, i.e. twice
as efficient per iteration); (3) a tile's **last** anti-diagonal
directly feeds the **first** anti-diagonal of the next tile (no gap),
improving cross-tile cache locality.

The cost: because tile columns are now tilted, the set of tiles active
in a given global iteration ("wavefront of tiles") differs from the
square-tile case, and **more** global iterations are needed overall
(increasing from `2*numTiles_x - 1` to `3*numTiles_x - 1` in the
chapter's worked example — a consequence of needing extra iterations
for the matrix's incomplete edge tiles under the tilted scheme). Figs.
16.15–16.19 give the full hypertile kernel implementation: `tile_col`'s
formula changes to reflect the tilt (`d - 2*blockIdx.x` instead of
`d - blockIdx.x`), the shear transform is applied via a `_shear()`
macro wherever a tile-local column index maps to a global-memory
column index, the `load_n`/`load_w`/`load_nw` device functions'
internals change to account for the sheared storage layout, and a new
`initialize_tile()` device function zero-fills shared memory first
(needed because, unlike square tiling, some threads in a hypertile are
always active even when their *assigned cell* is genuinely out of the
true scoring matrix's bounds — their writes are suppressed, but without
pre-zeroing, later loads could read uninitialized "garbage" shared
memory). Storing a parallelogram hypertile as a square array in shared
memory (row-major) also reintroduces a **bank-conflict** risk whenever
the hypertile width is a multiple/divisor of the bank count — solved,
as in Chapter 6, with **padding** (one extra memory location per row,
via a `pad()` macro).

(A sidebar extends the hyperplane idea beyond Smith-Waterman: Jacobi
**stencil computation**, Chapter 8, with an outer time-iteration loop
has both intra-tile wavefronts (within the spatial domain) and
inter-tile wavefronts (across time), and hypertiles can be applied
across the combined space-time domain to expose more parallelism and
reduce the number of required global synchronizations.)

## 16.8 More optimizations (p. 397)

**Synchronization across tile anti-diagonals**: kernel-termination-
based global sync (used throughout §16.6–16.7) forces every thread
block to wait for *all* blocks computing the previous two tile
anti-diagonals, even though it only truly depends on **three** specific
neighboring tiles — wasteful, causing load imbalance and wasted SM
slots. CUDA's **cooperative groups** API can synchronize an entire grid
without relaunching the kernel, but requires *persistent* thread blocks
sized to the device's actual concurrent-block capacity, and would still
force waiting on all of the previous two anti-diagonals' blocks — not
a finer-grained fix. A better mechanism: an **array of flags**, one per
tile, set when that tile's computation completes; a block needs only
check the (up to two) flags for the tiles directly above and to its
left, proceeding as soon as those are set (Fig. 16.21) — this closely
resembles the **unidirectional synchronization** (single lookback) used
for scan's single-kernel implementation (Chapter 11), and with
persistent thread blocks (one per tile *row*), only **one** flag check
is needed per block. The technique applies equally to hypertiles, where
hypertile width becomes a tunable knob trading off synchronization
overhead against per-block work.

**Small dynamic programming problems**: the chapter's examples assume
sequences large enough to need multiple thread blocks for one scoring
matrix. For short sequences (hundreds of base pairs), a **single
thread block or even a single warp** may suffice to compute an entire
scoring matrix — such implementations can benefit from keeping tiles in
**registers** and exchanging intermediate values via **warp shuffle**
instructions (Chapter 10) instead of shared memory.

**DPX instructions**: since the Hopper architecture, CUDA offers
specialized SIMD instructions (**DPX**) accelerating common dynamic-
programming primitives — e.g. `__vimax3_s32_relu()` can directly replace
the hand-written `max4()` device function (computing the max of three
values and 0 in specialized hardware) used throughout this chapter's
kernels.

## 16.9 Summary (p. 399)

Dynamic programming recursively decomposes complex problems into
simpler, overlapping sub-problems; bottom-up implementations tabulate
intermediate results, and the dependency structure between table cells
defines **wavefronts** — sets of sub-problems solvable in parallel,
which is the natural unit of GPU parallelism for this whole algorithm
class. The chapter showed how to exploit wavefront parallelism
efficiently via **block-level tiling** (reducing kernel-launch overhead
and improving data reuse via shared memory) and **hyperplane
(hypertile) transformation** (uniform per-iteration parallelism, fewer
iterations, better cross-tile cache locality) — with **inter-block flag-
based synchronization** and **DPX instructions** as further, more
advanced optimizations.

## 16.10 Exercises (p. 399)

Six implementation exercises: implement a square-block-tiled version of
the Floyd-Warshall kernel and compare it against §16.4's row-per-block
version (Q1); modify the Smith-Waterman block-tiling kernel to use
rectangular (rather than square) tiles (Q2); implement a version using
cooperative-groups grid-wide synchronization instead of kernel
termination (Q3); implement a version using §16.8's unidirectional
(single-lookback) flag-based synchronization, for both square tiles
(Q4) and hypertiles (Q5), referencing Chapter 11's single-lookback scan
as a model; and replace this chapter's `max4()` device function with an
equivalent DPX instruction (Q6). Not implemented in this repo's samples
(end-of-chapter exercises are out of scope — see the repo root README).
