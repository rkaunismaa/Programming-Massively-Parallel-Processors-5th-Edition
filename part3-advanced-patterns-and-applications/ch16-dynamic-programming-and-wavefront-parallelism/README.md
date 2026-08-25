# Chapter 16: Dynamic programming and wavefront parallelism

Source: *Programming Massively Parallel Processors*, 5th ed., Ch. 16 (pp. 373-400).

| File | Book section | Technique |
|------|-------------|-----------|
| `01_floyd_warshall.cu` | §16.4 | Bottom-up parallel Floyd-Warshall all-pairs shortest path: one kernel launch per intermediate vertex `k`, one thread per `(i, j)` distance cell (Fig. 16.4) |
| `02_smith_waterman.cu` | §16.5-16.6 | Basic (untiled) wavefront Smith-Waterman local sequence alignment: one kernel launch per anti-diagonal, one thread per scoring-matrix cell |
| `03_wavefront_block_tiled.cu` | §16.6 | Thread block-level tiling of Smith-Waterman's wavefront: one thread block per `threads x threads` tile, shared-memory anti-diagonal sweep, coalesced tile store-back (Figs. 16.7-16.12) |
| `04_hyperplane_transform.cu` | §16.7 | Hyperplane transformation (shear/hypertiles) applied to the same block-tiled Smith-Waterman kernel: uniform-length intra-tile wavefronts, fewer idle threads (Figs. 16.14-16.19) |

## Which kernel does §16.6's block-tiling apply to?

§16.6 ("Wavefront parallelization: block-level tiling") opens with: *"A basic
GPU parallelization of the **Smith-Waterman algorithm** launches one kernel
for each anti-diagonal wavefront..."*, and every figure in the section (host
code Fig. 16.8, kernel Fig. 16.9, device helpers Figs. 16.10-16.12) is built
around the Smith-Waterman scoring matrix `sw`/`H` (row/column indices `r`,
`q`, similarity score `subs_val`, `MATCH`/`MISMATCH`/`INSERTION`/`DELETION`).
Floyd-Warshall (§16.4) is never mentioned again after §16.4 itself -- its
wavefront already has full, constant size every iteration (§16.3: "similar
to examples (a) and (b) of Fig. 16.2"), so there is nothing for block-level
*sub*-tiling to exploit the way Smith-Waterman's grow/shrink diagonal
(Fig. 16.2(c)) benefits from it. `03_wavefront_block_tiled.cu` therefore
reimplements Smith-Waterman's `sw_kernel_square` tiled kernel, fully locally
(no `#include` of 01 or 02), per the task brief.

## §16.4 Floyd-Warshall -- `01_floyd_warshall.cu`

Reproduces Fig. 16.4's `FW_bottomup` kernel line for line: each thread owns
one `dist[row][col]` cell; thread 0 of the block stages `dist[row][k]` into
shared memory (`dist_k_col`), every thread reads `dist[k][col]` from global
memory (coalesced across the block), and updates its cell via
`min(dist[row][col], dist[row][k] + dist[k][col])`. The host (Fig. 16.4's
host snippet) launches one kernel per intermediate vertex `k`, synchronizing
via kernel termination between iterations. The "no edge" sentinel is renamed
`INF` (a plain int macro) since `<cmath>`, pulled in by `cuda_utils.h`,
already defines `INFINITY` as a float macro -- purely a naming fix, not a
logic change. The CPU reference is the textbook triple-nested-loop
Floyd-Warshall over the same recurrence. Distance matrices are compared for
**exact integer equality** (shortest-path distances have no rounding).

## §16.5-16.6 Basic wavefront Smith-Waterman -- `02_smith_waterman.cu`

§16.5 derives the scoring recurrence (Eq. 16.2) and shows (Fig. 16.6) that
each scoring-matrix cell depends only on its three neighbors in the two
previous anti-diagonals, so each anti-diagonal is an independent wavefront.
§16.6 then describes -- without a code figure -- the basic parallelization
this file implements: one kernel launch per anti-diagonal `d` (`r + q ==
d`), one thread per cell of that diagonal, computing
`max(0, H[r-1][q-1] + S(r,q), H[r-1][q] - insertion_penalty, H[r][q-1] -
deletion_penalty)`. The book states *"We leave the implementation of this
basic parallelization as an exercise"*, but (per this project's scope rule,
already applied the same way in `ch14-sorting`) the mechanism is fully
specified by the surrounding prose and Eq. 16.2, so it is implemented here
ahead of the tiled version. Two random DNA sequences (`ACGT` alphabet) of
equal length are generated per test case; the CPU reference fills the same
recurrence in row-major order (which happens to also respect the
dependency structure). Score matrices are compared for **exact integer
equality**.

## §16.6 Block-level tiling -- `03_wavefront_block_tiled.cu`

Reimplements Figs. 16.7-16.12 as printed: the scoring matrix is divided
into `threads x threads` tiles (Fig. 16.7(b)), one thread block per tile.
The host (Fig. 16.8) loops over *tile* anti-diagonals `d = 0 ..
2*numTiles_x-2`, launching `sw_kernel_square` once per `d` with
`threads*threads*sizeof(int)` bytes of dynamic shared memory. Inside the
kernel (Fig. 16.9), each thread sweeps its tile's own `2*tile_width-1`
anti-diagonals, loading the north/west/northwest neighbor cells via
`load_n`/`load_w`/`load_nw` (Fig. 16.10) -- from shared memory if the
neighbor is inside the tile, otherwise from global memory (produced by a
previous kernel call) -- taking `max4()` (Fig. 16.11) of the four
candidates, and `__syncthreads()`-ing between tile anti-diagonals. Once a
tile is fully computed, `store_tile` (Fig. 16.12) writes it back to global
memory row by row with all threads collaborating for coalesced accesses.
The book's template sequence-element type `T` is instantiated concretely as
`char` (the DNA alphabet used throughout §16.5); this and the CPU
reference/generator are duplicated locally per the file's self-containment
requirement -- it does not include `01_floyd_warshall.cu` or
`02_smith_waterman.cu`.

## §16.7 Hyperplane transformation -- `04_hyperplane_transform.cu`

Reimplements Figs. 16.15-16.19: a shear transformation with factor `m = -1`
(`_shear(x, y) = x + m*y`) turns each square tile into a parallelogram
"hypertile" where every column of the original tile becomes a uniform-length
anti-diagonal after the transform, cutting the intra-tile iteration count
from `2*tile_width-1` to `tile_width` and keeping all threads active in
every iteration (no more idle threads for short diagonals, per Fig. 16.13's
drawback). The host (Fig. 16.15) now loops `d = 0 .. 3*numTiles_x-2` (more
launches than the square-tile version, since the shear makes each thread
block two tile-columns ahead of the next, per §16.7's iteration-count
derivation). The kernel (Fig. 16.16) additionally calls `initialize_tile`
(Fig. 16.19) to zero the shared-memory tile first, since out-of-bound
threads stay active through every iteration and must not read garbage via
`load_n`/`load_w`/`load_nw` (Fig. 16.17, index math adjusted for the shear);
`store_tile` (Fig. 16.18) applies the shear when computing the global column
index. Because the recurrence itself is unchanged, the resulting score
matrix is byte-for-byte identical to the square-tile version's -- verified
here against the same CPU reference with exact integer comparison.

**Scope note:** §16.7's prose after Fig. 16.20 describes an additional
shared-memory *padding* optimization (a `pad(x)` macro) to avoid bank
conflicts, stating it would replace line 11 of Fig. 16.15's shared-memory
allocation. No updated kernel/device-function code with padded indexing
throughout is printed for it (unlike Figs. 16.15-16.19, which are complete
listings), so per this project's scope rule it is not implemented.

## Results

```
== bin/01_floyd_warshall ==
Parallel bottom-up Floyd-Warshall all-pairs shortest path (§16.4, Fig. 16.4):
n_vertices=1 edgeProb=0.50 threads=32: 10.6371 ms  [match]
n_vertices=5 edgeProb=0.50 threads=32: 0.0389 ms  [match]
n_vertices=37 edgeProb=0.30 threads=32: 0.2540 ms  [match]
n_vertices=128 edgeProb=0.15 threads=64: 0.8407 ms  [match]
n_vertices=200 edgeProb=0.05 threads=128: 1.3497 ms  [match]
PASS
== bin/02_smith_waterman ==
Basic wavefront Smith-Waterman: one kernel per anti-diagonal (§16.5-16.6):
L_seq=1     best_score(cpu)=3    best_score(gpu)=3    13.0499 ms  [match]
L_seq=2     best_score(cpu)=0    best_score(gpu)=0    0.0194 ms  [match]
L_seq=63    best_score(cpu)=52   best_score(gpu)=52   0.4898 ms  [match]
L_seq=256   best_score(cpu)=204  best_score(gpu)=204  1.9958 ms  [match]
L_seq=513   best_score(cpu)=379  best_score(gpu)=379  4.0134 ms  [match]
PASS
== bin/03_wavefront_block_tiled ==
Block-tiled wavefront Smith-Waterman (§16.6, Figs. 16.8-16.12):
L_seq=1     threads=32  best_score(cpu)=3    best_score(gpu)=3    0.4037 ms  [match]
L_seq=7     threads=32  best_score(cpu)=13   best_score(gpu)=13   0.0891 ms  [match]
L_seq=32    threads=32  best_score(cpu)=22   best_score(gpu)=22   0.2529 ms  [match]
L_seq=33    threads=32  best_score(cpu)=31   best_score(gpu)=31   0.4310 ms  [match]
L_seq=256   threads=32  best_score(cpu)=204  best_score(gpu)=204  3.6814 ms  [match]
L_seq=300   threads=64  best_score(cpu)=179  best_score(gpu)=179  4.1841 ms  [match]
PASS
== bin/04_hyperplane_transform ==
Hypertile (hyperplane-transformed) wavefront Smith-Waterman (§16.7, Figs. 16.15-16.19):
L_seq=1     threads=32  best_score(cpu)=3    best_score(gpu)=3    0.4611 ms  [match]
L_seq=7     threads=32  best_score(cpu)=13   best_score(gpu)=13   0.1116 ms  [match]
L_seq=32    threads=32  best_score(cpu)=22   best_score(gpu)=22   0.2898 ms  [match]
L_seq=33    threads=32  best_score(cpu)=31   best_score(gpu)=31   0.4762 ms  [match]
L_seq=256   threads=32  best_score(cpu)=204  best_score(gpu)=204  3.4191 ms  [match]
L_seq=300   threads=64  best_score(cpu)=179  best_score(gpu)=179  3.9544 ms  [match]
PASS
```

`L_seq=7, threads=32` in `03`/`04` covers the `L_seq < tile_width`
(single-tile, `numTiles_x == 1`) case -- see "Fix round 1" in the task-15
report for why this case matters for `04_hyperplane_transform.cu`'s
`store_tile`.

All four binaries also pass under `compute-sanitizer --tool memcheck` (0
errors) at `-arch=sm_75`.

Note: each binary's *first* printed timing (the `L_seq=1`/`n_vertices=1`
case) includes one-time CUDA context-initialization overhead and is not a
reliable measurement -- it can run ~10-50x slower than later, larger cases
timed within the same process. Compare timings across cases within a run,
not the first line in isolation.

Build and run all samples in this chapter:

```sh
make run
```
