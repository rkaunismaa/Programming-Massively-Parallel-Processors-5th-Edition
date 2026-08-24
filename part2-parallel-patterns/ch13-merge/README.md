# Chapter 13: Merge

Source: *Programming Massively Parallel Processors*, 5th ed., Ch. 13 (pp. 303-328).

| File | Book section | Technique |
|------|-------------|-----------|
| `01_merge_basic_parallel.cu` | §13.2, §13.4, §13.5 | Co-rank function (Fig. 13.5) + basic parallel merge kernel (Fig. 13.9) |
| `02_merge_tiled.cu` | §13.6 | Shared-memory tiled merge kernel to improve coalescing (Figs. 13.10-13.13) |
| `03_merge_circular_buffer.cu` | §13.7 | Circular-buffer tiled merge kernel -- full reuse of shared-memory data (Figs. 13.15-13.20) |
| `04_merge_coarsened.cu` | §13.8 | Thread coarsening: same kernel, `elementsPerThread=1` vs. `elementsPerThread=COARSE_FACTOR` |

All four files merge two sorted `int` arrays `A` (m elements) and `B` (n
elements) into a sorted output array `C` (m+n elements), and all use the
**same stability rule**, per §13.1: on a tie between an `A` element and a
`B` element, the `A` element is placed first. This is enforced by the
sequential merge's `A[i] <= B[j]` comparison (not `<`), and is exactly what
`co_rank()`'s exit condition encodes -- `A[i-1] <= B[j]` (equality
acceptable) but `B[j-1] < A[i]` (equality NOT acceptable). Every file's GPU
output is checked for **exact** equality against a CPU sequential-merge
reference (merge output is fully deterministic given this tie-breaking
rule), never just a multiset/sorted-order check.

## §13.2-§13.5 Sequential merge, co-rank, and the basic parallel kernel -- `01_merge_basic_parallel.cu`

**The co-rank function** (§13.4, Fig. 13.5) is the primitive every kernel in
this chapter is built on. `co_rank(k, A, m, B, n)` returns `i`, the unique
index such that `C[0..k-1]` (the output prefix of length `k`) is exactly
the merge of `A[0..i-1]` and `B[0..k-i-1]` -- the caller derives
`j = k - i`. It's a binary search: `i`/`j` are the current candidate
co-ranks, `i_low`/`j_low` bound the search from below (initialized to
`max(0, k-n)` / `max(0, k-m)` -- §13.4 explains this bound: at most `n`
elements of the output prefix can come from `B`, so at least `k-n` must
come from `A`). The loop's exit condition, `A[i-1] <= B[j]` AND
`B[j-1] < A[i]`, is exactly the stability rule above stated as a boundary
condition on the split point.

This file's `testCoRankBookExample()` unit test replays the book's own
worked example verbatim -- Exercise 1's arrays `A={1,7,8,9,10}` (m=5),
`B={7,10,10,12}` (n=4) -- and checks `co_rank()` against every value the
chapter text states outright: `co_rank(3,...) == 2` and
`co_rank(4,...) == 3` (Figs. 13.4, 13.6-13.8, thread 1's `k_curr`),
`co_rank(9,...) == 5` (Fig. 13.4, thread 1's `k_next`, both arrays fully
exhausted -- the `i>0`/`j<n` guards in the loop correctly short-circuit to
`i=m, j=n` immediately when `k=m+n`). It also checks the full
`merge_sequential()` output against Fig. 13.1's own stated properties: the
two tied 10s from `B` stay in relative order, and the tied 7 from `A`
precedes the tied 7 from `B`.

**The basic kernel** (§13.5, Fig. 13.9): each thread computes its output
slice `[k_curr, k_next)` from `elementsPerThread` and `tid`, calls
`co_rank()` twice (once for `k_curr`, once for `k_next`) to get the exact
input sub-array bounds, and calls `merge_sequential()` on its own slice.
No thread ever touches another thread's input or output range. §13.5
itself calls out why this is inefficient: neither the co-rank binary
searches nor the final sequential merge produce coalesced global-memory
accesses, since adjacent threads' input positions are wherever the
data-dependent search happened to land them. §13.6-13.7 fix this.

Measured (`nvcc -arch=sm_75 -Xptxas -v`): 38 registers/thread, 0 bytes
static shared memory.

## §13.6 Tiled merge kernel -- `02_merge_tiled.cu`

Implements Figs. 13.11-13.13 directly. Each **block** (not thread) first
calls `co_rank()` twice against the FULL global `A`/`B` arrays to get its
block-level input sub-array bounds (one thread does this, publishing the
result through shared memory) -- reducing the number of global-memory
binary searches from one-per-thread to one-per-block. The block then
iterates in `tile_size`-element chunks: each while-loop iteration
cooperatively, coalescedly loads up to `tile_size` elements of `A` and
`tile_size` elements of `B` into shared-memory arrays `A_S`/`B_S`
(consecutive threads load consecutive addresses), then every thread in the
block runs its own (much cheaper, shared-memory) `co_rank()` + merge over
that tile. §13.6 notes explicitly this wastes up to half the loaded data
every iteration: whatever fraction of the tile a thread's merge doesn't
consume is simply discarded and re-loaded fresh next iteration. §13.7 fixes
this with a circular buffer.

**Book-fidelity note on end-of-iteration bookkeeping**: §13.6's text states
the amount of `A` consumed by an iteration as
`co_rank(tile_size, A_S, tile_size, B_S, tile_size)` -- valid whenever both
tiles are genuinely full (`tile_size` elements each), true throughout the
book's own worked numeric example. This file instead passes the tile's
*actual* current valid lengths (already computed for that same iteration's
per-thread `co_rank()` calls) to this same bookkeeping call, which reduces
to the book's formula exactly when the tile is full, but also stays correct
when one input array runs out well before the other (exercised by this
file's "A empty" test case, `m=0`) -- with the literal `tile_size` bound,
that case reads uninitialized shared memory as if it were valid `A` data
and corrupts state used by a **non-final** iteration (caught by
`compute-sanitizer` during development, not just a logic-review finding).

Test cases include the book's own worked numbers verbatim: `m=33000`,
`n=31000`, `gridDim.x=16`, `blockDim.x=128`, `tile_size=1024` -- which
forces exactly the 4-iterations-with-a-partial-last-tile case (`4000`
output elements/block, `ceil(4000/1024)=4`, last iteration only `928`
elements) that §13.6's text calls out as the tricky boundary case.

Measured: 64 registers/thread, 0 bytes static shared memory (all of
`A_S`/`B_S` is dynamic shared memory, sized `2*tile_size*4` bytes at launch
-- `8192` bytes for the book's own `tile_size=1024` example, matching
§13.6's own stated `8,192` bytes exactly).

## §13.7 Circular-buffer merge kernel -- `03_merge_circular_buffer.cu`

Implements the circular-buffer scheme (Figs. 13.15-13.20): instead of
discarding a tile's unconsumed remainder every iteration, `A_S`/`B_S` keep
it in place and each new iteration only tops up the buffer with exactly as
many fresh elements as were consumed, wrapping the write position with
`% tile_size` (`A_S_start`/`B_S_start` track each buffer's current logical
start). Per §13.7's "simplified model" (Fig. 13.17b), `co_rank_circular()`
and `merge_sequential_circular()` use the *same* `i`/`j`/`i_low`/`j_low`
search logic as the plain `co_rank()`/`merge_sequential()` -- only the
array-access point is remapped through `(start + offset) % tile_size`, so
the circular nature of the buffer never leaks into the search/merge
algorithm itself.

**Book-fidelity disclosure, and a real bug this surfaced.** §13.7's text
gives the refill destination index as
`A_S_start+(tile_size-A_S_consumed)+i+threadIdx` where `A_S_consumed` is
"the number of A elements consumed" by the merge. That formula is only
correct when the tile was exactly `tile_size` full *before* that
consumption -- true throughout the book's own worked example, where both
`A` and `B` always have far more than `tile_size` elements remaining until
a block's very last iteration. Implementing that formula literally (first
attempt, during development) produced a genuine, reproducible mismatch on
this file's `book example` and `larger/random` multi-iteration test cases
(not the trivial one- or two-iteration cases), traced (by hand-simulating
one steady-state iteration) to duplicate loading: the refill's *source*
index (`A[A_curr+A_consumed+i]`, using cumulative *elements merged*) and
its *destination* offset (assuming *leftover = tile_size - consumed*) only
agree when nothing was ever left over short of a full tile. This file
instead tracks `A_loaded`/`B_loaded` (total elements ever fetched from
global memory, distinct from elements merged) and `leftoverA`/`leftoverB`
(elements currently resident but not yet merged) explicitly, requesting
exactly `tile_size - leftoverA` new elements (capped by what remains
globally) each iteration -- which reduces to the book's formula exactly
whenever the tile is genuinely full, and stays correct in general. Verified
against `compute-sanitizer` (0 errors) and exact CPU-reference match on all
four test cases, including the two multi-iteration ones that caught the
bug.

Test cases mirror file 02's exactly (same book-example numbers, same
larger/random case, same degenerate `tiny`/`A empty` cases) so the two
tiling strategies are directly comparable.

Measured: 49 registers/thread, 16 bytes static shared memory
(`blockCorank[2]` = 8 bytes + compiler padding) plus the same dynamic
`2*tile_size*4`-byte `A_S`/`B_S` buffer as file 02.

## §13.8 Thread coarsening for merge -- `04_merge_coarsened.cu`

§13.8's point is narrow: every kernel in this chapter is already
"coarsened" in the sense that `elementsPerThread` is normally `> 1` --
coarsening here isn't a separate algorithm, it's a launch-configuration
choice, and the book's stated motivation is amortizing each thread's two
`co_rank()` binary searches over more output elements. This file makes the
comparison concrete: the identical `mergeBasicKernel` from file 01, run
twice on the same input pair -- `elementsPerThread=1` ("uncoarsened", one
binary-search pair per output element) vs. `elementsPerThread=COARSE_FACTOR`
("coarsened") -- each with its own untimed warm-up launch immediately
before its own timed launch (this project's timing-fairness convention),
so both configurations do the exact same total merge work and the
comparison isolates thread count / binary-search count alone.

**Measured, not assumed** (RTX 4090, `-arch=sm_75`): `COARSE_FACTOR` was
chosen empirically. An initial attempt with `COARSE_FACTOR=64` measured
*slower* than uncoarsened (`~40 ms` vs. `~2 ms` under `compute-sanitizer`
instrumentation, and roughly a wash -- uncoarsened even slightly faster --
under plain timing): cutting the thread count 64x, down to the low
hundred-thousands for these problem sizes, left too few threads in flight
to keep this memory-bandwidth-bound kernel's memory pipeline saturated,
and that occupancy loss outweighed the binary-search savings.
`COARSE_FACTOR=8` (the value shipped in this file) keeps enough threads in
flight while still meaningfully cutting the total binary-search count, and
matches §13.8's claimed direction:

```
m=2000000 n=1500000 (total=3500000 output elements):
  uncoarsened   elementsPerThread=1  threads=3500000  0.1628 ms
  coarsened     elementsPerThread=8  threads=437500    0.1217 ms
m=4194304 n=4193527 (total=8387831 output elements):
  uncoarsened   elementsPerThread=1  threads=8387831  0.3881 ms
  coarsened     elementsPerThread=8  threads=1048479   0.2787 ms
```

roughly a 25-28% reduction at these sizes -- a real but modest win, not the
dramatic speedup a naive reading of §13.8 might suggest, because this
kernel remains memory-bandwidth-bound either way.

Measured: 38 registers/thread, 0 bytes static shared memory (same kernel
code as file 01).

## Results (RTX 4090, `sm_89`, binaries built for `-arch=sm_75`; this
environment's default CUDA device 0 is the 4090, not the 2070 SUPER)

```
== bin/01_merge_basic_parallel ==
Co-rank unit test (book's worked example, A={1,7,8,9,10} B={7,10,10,12}):
  co_rank(3, A,5, B,4) = 2 (expected 2)  [match]
  co_rank(4, A,5, B,4) = 3 (expected 3)  [match]
  co_rank(9, A,5, B,4) = 5 (expected 5)  [match]
  merge_sequential(A,5,B,4) = [1,7,7,8,9,10,10,10,12]  [match]

Basic parallel merge kernel (Fig. 13.9):
m=200000 n=150000 (threads=65536, elementsPerThread=6): 0.0143 ms  [match]
m=1048576 n=1036231 (threads=65536, elementsPerThread=32): 0.0891 ms  [match]
m=1 n=1 (threads=65536, elementsPerThread=1): 0.0040 ms  [match]
m=0 n=5000 (threads=65536, elementsPerThread=1): 0.0039 ms  [match]
PASS
== bin/02_merge_tiled ==
Tiled merge kernel (§13.6, Figs. 13.10-13.13):
book example: m=33000 n=31000 grid=16 block=128 tile_size=1024: 0.0307 ms  [match]
larger/random: m=500000 n=380000 grid=64 block=128 tile_size=512: 0.1116 ms  [match]
tiny: m=1 n=1 grid=1 block=128 tile_size=1024: 0.0061 ms  [match]
A empty: m=0 n=4000 grid=4 block=128 tile_size=512: 0.0061 ms  [match]
PASS
== bin/03_merge_circular_buffer ==
Circular-buffer tiled merge kernel (§13.7):
book example: m=33000 n=31000 grid=16 block=128 tile_size=1024: 0.0450 ms  [match]
larger/random: m=500000 n=380000 grid=64 block=128 tile_size=512: 0.1997 ms  [match]
tiny: m=1 n=1 grid=1 block=128 tile_size=1024: 0.0041 ms  [match]
A empty: m=0 n=4000 grid=4 block=128 tile_size=512: 0.0052 ms  [match]
PASS
== bin/04_merge_coarsened ==
Thread coarsening for merge (§13.8): same mergeBasicKernel, launched with
elementsPerThread=1 (uncoarsened) vs elementsPerThread=8 (coarsened).

m=2000000 n=1500000 (total=3500000 output elements):
  uncoarsened    elementsPerThread=1    threads=3500000    blocks=13672   0.1628 ms  [match]
  coarsened      elementsPerThread=8    threads=437500     blocks=1709    0.1217 ms  [match]
  -> uncoarsened (3500000 threads, 1 elem/thread) vs coarsened (437500 threads, 8 elem/thread): 0.1628ms vs 0.1217ms measured
m=4194304 n=4193527 (total=8387831 output elements):
  uncoarsened    elementsPerThread=1    threads=8387831    blocks=32765   0.3881 ms  [match]
  coarsened      elementsPerThread=8    threads=1048479    blocks=4096    0.2787 ms  [match]
  -> uncoarsened (8387831 threads, 1 elem/thread) vs coarsened (1048479 threads, 8 elem/thread): 0.3881ms vs 0.2787ms measured
PASS
```

All four binaries also pass under `compute-sanitizer` (0 errors) at
`-arch=sm_75`.

Build and run all samples in this chapter:

```sh
make run
```
