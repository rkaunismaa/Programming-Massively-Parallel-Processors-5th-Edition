# Chapter 13 — Merge (And Dynamic Input Data Identification)

Section-by-section summary of the book content. Page numbers are printed
book pages (Chapter 13 runs pp. 303–328).

Merge combines two sorted lists into one sorted list — a building block
for merge sort (Chapter 14) and for the reduce stage of map-reduce
frameworks. Its defining challenge, unlike every earlier pattern: **the
range of input elements each thread must consume depends on the actual
input *values*, not on a simple index formula** — making merge a genuine
"dynamic input data identification" problem, and the chapter's main
vehicle for teaching that class of problem.

## 13.1 Background (p. 303)

A merge function based on ordering relation `R` takes two sorted input
arrays `A` (size `m`) and `B` (size `n`), which need not be equal in
size, and produces one sorted output array `C` with all `m+n` elements
(Fig. 13.1). A merge is **stable** if, whenever `A` and `B` have equal
keys, their original relative order is preserved — both *within* one
input list (equal elements from the same list keep their order) and
*across* the two lists (an element of `A` with a tied key is placed
*before* a same-valued element of `B`, by the chapter's convention).
Stability matters because it lets a merge preserve ordering information
from an *earlier* sort key that the *current* merge's key doesn't
capture — essential for stable multi-key sorts. Merge is the core
operation inside **merge sort** (Chapter 14) and inside the **reduce**
phase of map-reduce frameworks (e.g. Hadoop), where many compute nodes'
partial results are commonly assembled via a merge-based reduction tree.

## 13.2 A sequential merge algorithm (p. 305)

Fig. 13.2's `merge_sequential`: a single `while` loop (indices `i`, `j`,
`k` into `A`, `B`, `C`) compares `A[i]` and `B[j]` each iteration,
copying the smaller (or `A[i]` on a tie, giving stability) into `C[k]`
and advancing that list's index plus `k`; once one list is exhausted,
two separate trailing `while` loops (only one of which actually runs)
copy the remainder of the other list. Each input element is visited
exactly once and each output position written exactly once — `O(m+n)`
work, linear in the combined list size.

## 13.3 A parallelization approach (p. 306)

Following Siebert et al.'s approach: each thread first determines the
**range of output positions** it will produce, then uses a **co-rank
function** to map that output range to the corresponding input ranges
in `A` and `B` — once input/output ranges are known, each thread
independently runs the sequential merge function on its own
sub-arrays, fully in parallel. The co-rank function is the crux of the
whole approach.

**Observation 1**: for any output rank `k` (0 ≤ k < m+n), `C[k]` is
filled from either some `A[i]` or some `B[j]` in the merge process.
**Observation 2** (generalizing via symmetry, illustrated in Fig.
13.3's two worked cases): for any `k`, there exist unique `i`, `j` with
`k = i+j` such that `C[0..k-1]` is exactly the result of merging
`A[0..i-1]` and `B[0..j-1]` — `i` and `j` are called `k`'s **co-ranks**.
Siebert et al. proved these co-ranks are unique for a given `k`. Once
the output array is divided into per-thread subarrays by rank (a simple
index calculation, same as every earlier pattern), each thread calls
the co-rank function to translate its *output* rank into the
*corresponding input subarrays* it needs to merge — this translation
step is what's genuinely new and challenging about parallelizing merge,
versus every pattern covered so far.

## 13.4 Co-rank function implementation (p. 308)

Signature: `co_rank(k, A, m, B, n)` returns `i`, the co-rank of `k` in
`A` (the caller derives `j = k - i`). Fig. 13.4 walks a 2-thread,
9-element example by hand, confirming by direct inspection that only
one `(i,j)` split keeps the merge correctly ordered — shifting `i`
either up or down breaks sortedness, motivating a **search** that can
find the right split efficiently.

Since both inputs are sorted, **binary search** gives `O(log N)`
co-rank computation (Fig. 13.5): `i`/`j` are the current candidate
co-ranks (initialized so `i = min(k,m)`, `j = k-i`); `i_low`/`j_low`
track the smallest feasible co-rank values (initialized to
`max(0, k-n)` and `max(0, k-m)` respectively, since at most `n`/`m`
elements can come from the *other* array). The loop's invariant is
`i + j == k` throughout. Each iteration checks whether the current
split is already correct (`A[i-1] <= B[j]` and `B[j-1] < A[i]`,
matching Observation 2 plus the stability tie-break); if `i` is too
high (`A[i-1] > B[j]`), it's roughly halved toward `i_low` (and `j`
correspondingly raised); if `j` is too high (symmetric condition on
`B`/`A`), the opposite adjustment is made. Figs. 13.6–13.8 trace all
three iterations of this search for a concrete 3-thread example,
confirming it converges on the unique correct co-ranks. The search
is `O(log N)` where `N` is the larger of the two input sizes, since the
search range roughly halves each iteration regardless of which branch
fires.

## 13.5 A basic parallel merge kernel (p. 313)

Fig. 13.9's `merge_basic_kernel`: each thread computes `k_curr` and
`k_next` (its own and the next thread's starting output rank, handling
a non-evenly-divisible total via `elementsPerThread = ceil((m+n)/
(blockDim.x*gridDim.x))`), calls `co_rank` **twice** (once for
`k_curr`, once for `k_next`) to get `i_curr`/`i_next` (and derives
`j_curr`/`j_next` as `k_curr - i_curr`/`k_next - i_next`), then calls
`merge_sequential` directly on its own slice of `A`/`B`/`C`. Traced
through the running 3-thread example (Fig. 13.8): thread 1 ends up
correctly merging `A[2..4]` with (zero elements of) `B[1..0]` into
`C[3..5]` — matching the hand-worked example exactly.

This kernel is simple but memory-inefficient in two independent ways:
(1) **output/input element accesses within `merge_sequential` are not
coalesced** — adjacent threads' `i`/`j`/`k` values are generally far
apart (e.g. threads 0, 1, 2 simultaneously read `A[0]`, `A[2]`, `B[0]`
and write `C[0]`, `C[3]`, `C[6]` in the chapter's example) since each
thread's assigned input range depends on data values, not a uniform
stride; (2) **the co-rank function's own binary search accesses are
themselves irregular/uncoalesced**, compounding the problem.

## 13.6 A tiled merge kernel to improve coalescing (p. 314)

Of Chapter 6's three general coalescing strategies (rearrange
thread-to-data mapping; rearrange the data itself; stage data through
shared memory in a coalesced pattern), merge uses the third: since each
block's threads collectively use input ranges of `A`/`B` that are
*adjacent in memory* even though individual threads' ranges aren't
uniform, the **block as a whole** can call `co_rank` once to get its
own larger block-level input range, then **cooperatively**, coalesced-ly
load that whole range into shared memory (`A_S`/`B_S`) before
individual threads divide up the now-on-chip data (Fig. 13.10) — a
secondary benefit is reused data (both the co-rank searches and the
merge itself) now being served from fast shared memory instead of
repeated global-memory round trips.

Because the shared-memory tiles (`A_S`, `B_S`) can't necessarily hold
an entire block's whole input range at once, the kernel is **iterative**:
each iteration loads up to `tile_size` elements of each input array,
generates `tile_size` elements of output, then loads the next
`tile_size`-sized chunk. Three-part kernel (Figs. 13.11–13.13): **Part
1** — one thread computes the block's own `C_curr`/`C_next` boundaries
and calls `co_rank` (block-level, against *global* `A`/`B`) to get
`A_S[0]`/`A_S[1]` holding the block's own `A_curr`/`A_next`, shared via
shared memory + barrier. **Part 2** — the whole block cooperatively,
coalesced-ly loads up to `tile_size` elements each of `A`/`B` into
`A_S`/`B_S` each iteration, using an `if` guard (checking against
`A_length - A_consumed` / `B_length - B_consumed`) so a block near the
end of its assigned range doesn't load past its own boundary. **Part
3** — each thread computes its own output sub-range within the current
`tile_size`-sized iteration, calls `co_rank` **against the on-chip
tiles** (not global memory) to split that work further, and calls
`merge_sequential` on the shared-memory data; `A_consumed`/`B_consumed`/
`C_completed` are then updated for the next iteration.

Worked numeric example: merging 33,000- and 31,000-element arrays
(64,000 total output) with 16 blocks × 128 threads, `tile_size=1024`
(8,192 B of shared memory per block) needs 4 while-loop iterations per
block, each loading 1,024 elements of `A` and `B` (128 threads loading 8
elements each, per for-loop) — with 2,048 total elements loaded but
normally only 1,024 actually consumed in that iteration (varies by data
values — all 1,024 consumed elements could even come from just one of
the two arrays). This surfaces the kernel's remaining deficiency: **only
half of the loaded tile data gets used per iteration on average**, with
the unused remainder simply reloaded (redundantly) next iteration —
wasting roughly half the achievable memory bandwidth, motivating §13.7.

## 13.7 A circular-buffer merge kernel (p. 321)

Fixes §13.6's waste by tracking, per block, a **circular buffer**
`A_S_start`/`B_S_start` offset into each shared-memory tile, so that
only the genuinely *unconsumed* remainder from the previous iteration
gets kept, and each new iteration loads just enough fresh elements
(`A_S_consumed`/`B_S_consumed`-sized) to refill the tile back to
`tile_size` — reusing, rather than discarding and reloading, whatever
wasn't consumed last time (Fig. 13.15 walks four iterations of this
wraparound by hand). The modulo (`%tile_size`) arithmetic is what makes
this a genuine *circular* buffer: both the fill region and the
consumed region can wrap around the physical array.

To keep the co-rank search and sequential-merge code itself simple
despite this added complexity, the chapter introduces a **simplified
index model** (Fig. 13.17): `co_rank_circular` and
`merge_sequential_circular` take the same parameters as the originals
plus `A_S_start`, `B_S_start`, `tile_size`, and internally present the
*illusion* that each tile is one contiguous region starting at
`A_S_start`/`B_S_start` (so `a_next >= a_curr` always *appears* true to
the caller, even when the real underlying buffer has wrapped) — all the
actual wraparound bookkeeping (computing real indices like
`(A_S_start + i) % tile_size`) is pushed down into these two low-level
functions (Figs. 13.19–13.20), while the higher-level kernel logic
(Fig. 13.18) stays nearly identical to the non-circular version. This
illustrates a general software-engineering lesson: a well-designed
library interface can absorb a sophisticated data structure's
complexity so calling code doesn't have to change much.

A noted tradeoff: circular-buffer management needs several extra
per-thread registers (tracking buffer start/consumed counts), which can
reduce SM occupancy — but since merge is fundamentally memory-bandwidth-
bound rather than compute-bound, trading some register/compute headroom
for better memory bandwidth utilization is judged a reasonable
tradeoff.

## 13.8 Thread coarsening for merge (p. 327)

Merge's parallelization overhead comes mainly from every thread needing
to run its own `co_rank` **binary search** — so a fully uncoarsened
kernel (one output element per thread) would pay a binary search per
*single* element, prohibitively expensive. Every kernel actually
presented in this chapter is **already coarsened** (each thread
produces multiple output elements), which amortizes each thread's
binary-search cost across a substantial number of output elements —
coarsening here isn't an optional extra optimization layered on top (as
in earlier chapters) but a load-bearing design requirement from the
start.

## 13.9 Summary (p. 327)

Merge is the chapter's vehicle for a genuinely new class of problem:
parallelizing a pattern whose per-thread input ranges are **data-
dependent** rather than index-computable. The **co-rank function**
(binary search over sorted inputs) is the key tool for resolving this.
Making the tiled, coalescing-oriented version work required a
**circular buffer** to avoid wasting half the loaded shared-memory
data — and introducing that more complex data structure in turn
motivated a **simplified access model** so the complexity stays
contained inside a small set of buffer-aware helper functions, leaving
the higher-level kernel code largely unchanged.

## 13.10 Exercises (p. 327)

Problems include: computing co-rank values for `C[8]` given two
specific 5- and 4-element lists (Q1); completing the co-rank
calculation for "thread 2" in the chapter's running 3-thread example
(Q2, left incomplete in the main text as an exercise for the reader);
modifying the tiled kernel's load loops to call `co_rank` so only the
elements actually needed for the current iteration are loaded, rather
than a full tile (Q3); and a multi-part numeric question computing how
many threads perform binary search on global- vs. shared-memory data,
across the basic kernel and the tiled kernel, for a specific large
merge configuration (Q4). Not implemented in this repo's samples
(end-of-chapter exercises are out of scope — see the repo root README).
