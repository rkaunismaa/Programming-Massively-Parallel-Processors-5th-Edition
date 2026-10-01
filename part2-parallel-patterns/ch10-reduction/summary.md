# Chapter 10 — Reduction (And Mitigating Control Divergence)

Section-by-section summary of the book content. Page numbers are printed
book pages (Chapter 10 runs pp. 221–250).

## 10.1 Background (p. 221)

A **reduction** derives a single value from a list, using a binary
operator — sum, product, min, max, etc., over integers, floats of
various precisions, or other types. Reduction is an important pattern
(it distills a summary from large data) and, unlike convolution/stencil,
requires **all threads to coordinate** to produce one final answer,
making it a natural vehicle for the chapter's bottleneck-mitigation
techniques. A sequential sum reduction (Fig. 10.1) initializes an
accumulator to the operator's **identity value** (0.0 for addition) and
loops, applying the operator between the accumulator and each element in
turn; Fig. 10.2 generalizes this to any binary `Operator`. Min/max
reductions use `+∞`/`-∞` as their respective identities. (CUDA/C++
libraries — Thrust, CUB, `std::reduce` — already provide optimized
reductions; the chapter builds one from scratch purely as a vehicle for
teaching parallelization/optimization technique.)

## 10.2 Reduction trees (p. 223)

Parallelizing reduction means **reordering** the sequence of operator
applications: a **reduction tree** (Fig. 10.3) applies the operator to
disjoint pairs in parallel each round, halving the number of remaining
values every round, until one value remains. This reordering is only
valid if the operator is **associative** (`(a⊖b)⊖c = a⊖(b⊖c)` — true for
integer/real addition, max, min; not true for subtraction, and not
*exactly* true for IEEE floating-point addition due to rounding, though
many applications tolerate the resulting small discrepancy). A further
optimization (§10.4) also reorders the *operands* themselves, which
additionally requires the operator to be **commutative** (`a⊖b = b⊖a` —
true for addition/min/max, not for subtraction). A reduction tree is
conceptually a tree only in its information-flow pattern (parent/child
edges are not an actual pointer-linked data structure).

The chapter introduces **work** (total operations performed) and
**span** (time needed assuming unlimited execution resources, i.e. the
number of *sequential* steps) as the key efficiency metrics, noting that
sequential algorithms always have work = span, while parallel
algorithms can decouple the two. A reduction tree over `N` elements
performs the same `O(N)` work as the sequential version (specifically
`N-1` operations: `½N + ¼N + ⅛N + ... = N-1`, a geometric series) — i.e.
it does not change the total work — but has span `O(log N)` instead of
`O(N)` (10 steps for `N=1024`, vs. 1024 sequentially), since each step
halves the active count. This also means the *parallelism* required
varies sharply across steps — peak 512 execution units at step 1 for
`N=1024`, dwindling to 1 at the final step, averaging only 102.3 — which
the book notes makes reduction trees a uniquely challenging pattern for
hardware resource utilization (illustrated with a real-world analogy:
single-elimination sports tournaments, e.g. the 2010 World Cup, are
reduction trees whose "resources" are stadiums/cities, and sharing
limited resources across rounds similarly forces some rounds to spread
out over multiple days).

## 10.3 A simple reduction kernel (p. 227)

Because barrier synchronization (`__syncthreads()`) only works within
one block, a first implementation (Fig. 10.5) is restricted to a
**single thread block** reducing up to `2×1024 = 2048` elements (lifted
in §10.9). Thread `t` is assigned (owns) `input[2t]` (i.e.
`i = 2*threadIdx.x`); a loop over `stride = 1, 2, 4, ...` has each
*active* thread (`threadIdx.x % stride == 0`) add its stride-away
neighbor into its own owned location, following the **owner-computes**
rule (Fig. 10.6 traces this for 16 elements/8 threads). A
`__syncthreads()` after each iteration ensures all writes from the
current round are visible before any thread reads for the next round
(also implicitly serving as the required memory fence, since reader and
writer are in the same block). After `log₂(blockDim.x)` iterations,
thread 0's location holds the final sum, written to `*output`.

## 10.4 Reducing control divergence (p. 230)

Fig. 10.5's thread-to-location assignment causes severe **control
divergence**: the active-thread condition (`threadIdx.x % stride == 0`)
is satisfied by a shrinking, increasingly *scattered* subset of each
warp's threads each iteration — e.g. only 1 of 32 threads active by the
6th iteration, wasting 31/32 of that warp's execution resources even
though the *whole warp* is still consuming them. A full numeric analysis
for `N=256` gives an **execution resource utilization of only 30%**
(255 useful committed results out of 864 total resources consumed).

The fix (Fig. 10.7) reassigns threads so that **stride shrinks instead
of grows** over time: thread `t` owns `input[t]` directly
(`i = threadIdx.x`), the loop's `stride` starts at `blockDim.x` and is
halved each iteration, and the active condition becomes simply
`threadIdx.x < stride` — so active threads are always a *contiguous,
leading* block of thread indices, meaning entire warps drop out cleanly
together rather than becoming partially active. This still requires a
*commutative* operator too, since it changes which operands get paired
with which (effectively reordering the input list itself, not just
re-parenthesizing it). Fig. 10.8's `ConvergentSumReductionKernel`
implements this; the same 256-element analysis now gives **66%
utilization** (255/384) — almost double — though divergence isn't fully
eliminated (once active-thread count drops below 32, that single warp
still has only some lanes active).

## 10.5 Reducing memory access divergence (p. 234)

Fig. 10.5's access pattern is also poorly **coalesced**: a warp's
owned locations are `stride` elements apart, so each round's reads/write
span widely separated addresses, triggering far more memory
transactions than necessary and wasting most of the returned data (e.g.
2× the needed transactions in round 1, growing each round). Fig. 10.8's
reassignment fixes this too, as a side effect of making active threads
contiguous: adjacent threads now always access *adjacent* memory
locations, so every active warp's accesses are **fully coalesced**,
minimizing both transaction count and DRAM bandwidth waste — a second,
independent benefit layered on top of the control-divergence fix.

## 10.6 Reducing global memory accesses (p. 236)

Fig. 10.8 still round-trips every intermediate partial sum through
**global memory** each iteration (read, add, write-back), which even
with caching carries significant latency/bandwidth cost. Fig. 10.10
instead stages the block's working data in **shared memory**: each
thread loads and adds its two original elements into shared memory once
(the first round, done "for free" outside the loop), then all further
rounds read/write only shared memory, with `__syncthreads()` still
required each round (Fig. 10.9). This drops total global memory traffic
to just `N+1` requests regardless of iteration count (reduced further to
`(N/32)+1` with coalescing) — a measured **4× improvement** (36 → 9
requests) in the chapter's running 256-element example. A secondary
benefit: the original input array is left unmodified, useful if other
parts of a program still need it.

## 10.7 Reducing synchronization overhead with warp-level primitives (p. 237)

Once the active-thread count drops to 32 (one warp), Fig. 10.10 is still
paying a full `__syncthreads()` + shared-memory round trip per
iteration — wasted overhead, since a single warp's threads can
communicate directly through **warp-level primitives** without shared
memory or barriers at all. `__syncwarp()` is the warp-scope analogue of
`__syncthreads()` (also forces warp reconvergence after divergence).
**Warp shuffle functions** let threads in a warp exchange register
values directly; `__shfl_down_sync(mask, var, delta)` (Fig. 10.11)
returns, for thread `i`, the value of `var` held by thread `i+delta`
(the `mask` argument marks which lanes participate).

Fig. 10.13's `warp_reduce` device function implements a complete
warp-wide sum reduction purely with `__shfl_down_sync` in a `stride =
16, 8, ..., 1` loop — no shared memory, no barriers — leaving the true
sum in lane 0's `partialSum`. Two helper device functions,
`warpIdx()` (`threadIdx.x / WARP_SIZE`) and `laneIdx()`
(`threadIdx.x % WARP_SIZE`), identify a thread's warp and its position
within it. Fig. 10.14's kernel runs Fig. 10.10's shared-memory loop only
down to `stride == WARP_SIZE`, then hands off to `warp_reduce` for the
remaining warp (warp 0 only, once reduced to ≤32 active elements) —
eliminating shared-memory access and barrier overhead for the loop's
final 5 iterations. (`__shfl_down_sync` also takes an optional *width*
parameter to subdivide a warp into smaller independent shuffle groups;
other primitives — `__shfl_sync`, `__shfl_up_sync`, `__shfl_xor_sync`,
warp voting functions, warp match functions, warp reduce functions — are
named but deferred to the CUDA Programming Guide / later chapters, e.g.
warp voting functions in Chapter 12.)

## 10.8 Further reducing synchronization overhead with two-stage warp-wide reduction (p. 241)

Fig. 10.14 still pays shared-memory/barrier overhead for its *early*
iterations (before the active set shrinks to one warp). The fix: have
**every** warp in the block perform its own independent warp-wide
reduction *first* (Fig. 10.15), each ending with one partial sum held by
that warp's lane 0; those per-warp partial sums (one per warp, placed by
lane 0 into shared memory, a single `__syncthreads()`) are then combined
by a *second* warp-wide reduction run by just the first warp. Fig.
10.16's `TwoStageWarpLevelSumReductionKernel`: each thread adds its two
elements and immediately calls `warp_reduce` (line 05, all warps, no
barrier); each warp's lane 0 writes its partial sum to
`partialSums_s[warpIdx()]`; one `__syncthreads()`; warp 0 loads these
partial sums and does a second `warp_reduce`, writing the final result.
This eliminates shared-memory access/barrier overhead from the *entire*
first stage (not just its tail), at the cost of reintroducing some
control divergence within each warp's shuffle operations during that
stage — a worthwhile tradeoff since, once data is loaded, performance is
dominated by instruction *latency* rather than by divergence-driven
resource underutilization.

## 10.9 Reduction for arbitrary length inputs (p. 244)

All prior kernels are capped at a single block's `__syncthreads()`-able
scope (≤2048 elements). Scaling to millions/billions of elements needs
**multiple independently-executing blocks** (Fig. 10.17): the input is
partitioned into per-block segments, each block reduces its own segment
to a partial sum, and all blocks accumulate their partial sums into the
final output via a **device-scope atomic add**. Fig. 10.18 adapts Fig.
10.16 for this: `segment = 2*blockDim.x*blockIdx.x` locates each block's
region (so the rest of the per-block reduction code is unchanged), and
the final write (previously a plain store by thread 0) becomes an
atomic `fetch_add` into `*output`. Since atomics only guarantee mutual
exclusion, not ordering, blocks may commit in any order — requiring the
operator to be **both commutative and associative** for correctness at
this scale (not just associative, as a single-block tree needed).
Alternatives mentioned (not worked out in code): have each block's
leader write its partial sum into an array indexed by block, then launch
a second, single-block kernel to finish the reduction; or copy partial
sums to the host and finish on the CPU — both useful when atomics are
unavailable/expensive, or when deterministic ordering of accumulation is
wanted.

## 10.10 Thread coarsening to reduce overhead (p. 246)

Maximizing parallelism (one thread per element pair, `N/2` threads
total) pays a real cost: every stage of the reduction tree
*underutilizes* the hardware more than the last (fewer and fewer active
threads/warps), and this underutilization recurs for *every* block the
hardware must serialize if there isn't enough parallel capacity to run
them all at once — an inherent, unavoidable price of actually running
blocks in parallel, but pure waste if the hardware would have serialized
them anyway. **Thread coarsening** (Chapter 6) addresses this directly:
give each thread more elements to reduce *independently* before joining
the tree-based phase, shrinking the block count for serialized
scenarios.

Fig. 10.19 extends Fig. 10.9's example with a coarsening factor of 2:
each block now receives 32 elements (4 per thread, up from 2), and each
thread independently sums its 4 elements (3 sequential adds, no
synchronization needed since each thread's elements are disjoint) before
the shared tree-reduction phase proceeds as before. Fig. 10.20's
`CoarsenedSumReductionKernel` changes only the segment-size
scaling (`COARSE_FACTOR*2*blockDim.x`) and replaces the single add with
a `COARSE_FACTOR*2`-iteration coarsening loop (no barriers needed, since
threads act independently in this phase). Fig. 10.21 compares, for a
coarsening factor of 2: two original (uncoarsened) blocks serialized by
hardware take 8 total steps (2 fully-utilized, 6 underutilized +
needing barriers/shared memory), versus one coarsened block doing the
same total work in 6 steps (3 fully-utilized, only 3 underutilized) —
strictly fewer underutilized/synchronization-heavy steps for the same
work. Coarsening factor can't grow unboundedly, though: too high a
factor launches too few blocks to fill the hardware's actual parallel
capacity, trading away real parallelism for no further benefit — so the
best factor is dataset- and device-specific.

## 10.11 Summary (p. 249)

Parallel reduction is an important pattern underlying many applications
(and foundational for the next chapter's prefix-sum/scan pattern), but
achieving high performance from it requires a progression of techniques
covered here: divergence-aware thread-to-data assignment, shared memory
to cut global memory round trips, warp-level primitives to cut
synchronization/shared-memory overhead, atomic-operation-based
multi-block reduction for arbitrary input sizes, and thread coarsening
to cut parallelization overhead when hardware can't actually run
everything concurrently. In practice, production code should reach for
already-optimized library implementations (Thrust, CUB) rather than
hand-rolling reduction — the value here is pedagogical, illustrating
general parallelization/optimization technique.

## 10.12 Exercises (p. 249)

Problems include: counting divergent warps at the 5th iteration for the
simple kernel (Fig. 10.5, Q1) and the improved kernel (Fig. 10.8, Q2);
modifying Fig. 10.8 to use a different (diagrammed) access pattern (Q3);
adapting the coarsened kernel (Fig. 10.20) to max reduction (Q4) and to
arbitrary-length (non-multiple-of-segment-size) input with an added `N`
parameter (Q5); and hand-tracing array contents after each iteration for
a given 8-element input, under both the unoptimized kernel (Fig. 10.5)
and the coalescing/divergence-optimized kernel (Fig. 10.8) (Q6). Not
implemented in this repo's samples (end-of-chapter exercises are out of
scope — see the repo root README).
