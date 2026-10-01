# Chapter 11 — Scan (And Work Efficiency in Parallel Algorithms)

Section-by-section summary of the book content. Page numbers are printed
book pages (Chapter 11 runs pp. 251–287).

Scan parallelizes operations that look inherently sequential — resource
allocation, work assignment, polynomial evaluation — and is itself used
as a primitive inside filtering (Chapter 12) and radix sort (Chapter
14). The chapter's throughline is **work efficiency**: some parallel
algorithms do strictly more total work than their sequential
counterpart, and the chapter uses scan to show both why that happens
and how to claw the extra work back when hardware resources are scarce.

## 11.1 Background (p. 251)

An **inclusive scan** with binary operator `⊕` takes `[x0, x1, ...,
xN-1]` and returns `[x0, x0⊕x1, ..., x0⊕x1⊕...⊕xN-1]` — each output
*includes* the corresponding input element. With addition, this is also
called **prefix sum**. Worked analogy: cutting a 40-inch sausage among 8
people who each ordered a different length — an inclusive scan over the
ordered lengths directly gives every cut point, so all cuts can be made
in parallel (or any order) once the scan result is known. An
**exclusive scan** instead returns `[ID⊕, x0, x0⊕x1, ..., x0⊕...⊕xN-2]`
— each output *excludes* its own input element, with the first output
being the operator's identity value; this gives each section's
*starting* point rather than its ending point, useful for e.g. memory
allocation offsets. Converting between the two is a simple shift (shift
right + fill identity for inclusive→exclusive; shift left + append
`last⊕final_input` for exclusive→inclusive); the chapter works
exclusively with inclusive scan for concreteness. A sequential inclusive
scan (Fig. 11.1) is a single `O(N)` loop accumulating into `output[i] =
output[i-1] + input[i]`.

## 11.2 Parallel scan with the Kogge-Stone algorithm (p. 254)

A naive "one thread per output, each doing its own reduction" approach
is a trap: output `y[N-1]` alone needs `N` sequential additions, so the
*span* is still `O(N)` — no faster than sequential — while the *work*
balloons to `Σi = O(N²)` (far worse than sequential's `O(N)`), since
each output's reduction duplicates most of the previous output's work
with nothing shared between them.

The **Kogge-Stone algorithm** (from 1970s fast-adder-circuit design)
fixes this by sharing partial sums across outputs. It's an **in-place**
algorithm: after `k` iterations, position `i` holds the sum of up to
`2^k` elements ending at `i` (Fig. 11.2, 16-element example). Fig.
11.3's `scan_kernel`: each thread loads its element into shared
`buffer_s`; a loop with `stride = 1, 2, 4, ...` has each thread whose
index `>= stride` add its stride-away left neighbor into a `temp`
variable, synchronize, then write `temp` back — needing **two**
barriers per iteration (unlike Chapter 10's reduction tree), because
unlike reduction, a scan's intermediate *positions* are read by *other*
active threads in the same iteration, creating both a true
(read-after-write) and a false (write-after-read) dependence that must
each be enforced; the `temp` variable plus the second barrier prevents
one thread's early write from corrupting another thread's still-pending
read of the old value (traced through a concrete 2-thread example where
omitting the second barrier silently produces a wrong, run-dependent
result). Control divergence from this loop is modest for wide blocks
(only the first warp exhibits it, and only while `stride` is smaller
than warp size).

## 11.3 Double-buffering to reduce synchronization (p. 258)

The second `__syncthreads()` exists purely to prevent a *false*
(write-after-read) dependence — exactly the kind Chapter 6 showed can be
eliminated with **double buffering**. Fig. 11.4/11.5: keep two shared
buffers (`buffer1_s`, `buffer2_s`) and two pointers (`inBuffer_s`,
`outBuffer_s`) that swap roles each iteration — reads always come from
the frozen "in" buffer while writes land in the separate "out" buffer,
so no thread can ever read a value another thread has already
overwritten this iteration. This leaves only **one** `__syncthreads()`
per iteration (enforcing the true dependence only), cutting
synchronization overhead roughly in half versus Fig. 11.3.

## 11.4 Warp-level primitives to reduce synchronization (p. 261)

Even with double buffering, every iteration still pays a full
barrier-plus-shared-memory round trip. The fix follows Chapter 10's
playbook: decompose the computation so warps can use warp-level
primitives for most of it. The general tool here is the
**scan-scan-add decomposition** (Fig. 11.6): scan each segment
independently (stage 1), scan the array of per-segment *totals* (stage
2), then add each segment's corresponding scanned-total value to every
element of the *next* segment (stage 3, since a segment's own total is
excluded from what's added to its own elements).

Applied at the block level (Fig. 11.7): each warp does its own
warp-level Kogge-Stone scan on its sub-segment (`warpScan`, Fig. 11.8 —
built from `__shfl_up_sync` instead of shared memory, so a thread
obtains its stride-away left neighbor's value directly from that
thread's register); each warp's last thread writes its warp's total sum
to a small shared array; a single warp then warp-scans that array of
warp-sums; finally every (non-first-warp) thread adds the appropriate
scanned warp-sum to its own value. Fig. 11.9's `blockScan` device
function and Fig. 11.10's resulting kernel need only **two**
barrier/shared-memory touch-points total (between stage 1 and stage 2,
and between stage 2 and stage 3) rather than one per loop iteration.

## 11.5 Work efficiency considerations (p. 265)

**Work efficiency** measures how close an algorithm's total work comes
to the theoretical minimum (here, `N-1` additions, matching the
sequential algorithm). The Kogge-Stone approach performs
`N·log₂N - (N-1)` additions (derived by summing active-thread counts
across all `log₂N` iterations) — in practice closer to `N·log₂N`, since
inactive threads in an active warp still consume execution resources.
This is strictly worse than sequential's `O(N)` (e.g. 8–9× more work at
N=512), even though its *span* (`O(log N)`) is far better — a direct
illustration that more parallel ≠ less total work, and that with only
`P < N` execution units available, the step count becomes roughly
`N·log₂N / P`, which can exceed the sequential step count `N` for small
`P`. This has two costs beyond slower execution with limited resources:
wasted hardware provisioning, and wasted energy — a real concern for
power-constrained (e.g. mobile) deployments.

## 11.6 Coarsening to improve work-efficiency (p. 267)

Thread coarsening recovers work efficiency by having each thread run a
**sequential** (hence `O(N/P)`-work, work-efficient) scan over its own
subsegment first, then having only the `P` per-thread totals
participate in the (work-*inefficient* but now much smaller) parallel
scan stage, with a final add-pass folding each preceding thread's
scanned total back into its subsegment — the scan-scan-add
decomposition again, now applied *across threads* rather than warps
(Fig. 11.11). Fig. 11.12's kernel: loads each block's segment into
shared memory using **coalesced** chunked loads (not naive
contiguous-per-thread loads, which would be strided and
uncoalesced — each of `COARSE_FACTOR` rounds has thread `t` load
element `c*BLOCK_DIM + t`); each thread then does a sequential scan of
its own `COARSE_FACTOR`-sized subsegment; the per-thread final values
feed a `blockScan` (Fig. 11.9); each thread adds the preceding thread's
scanned total to its whole subsegment; results are written back with
the same coalescing-preserving chunked pattern.

Work analysis: with `P` threads scanning `N` elements, per-thread
sequential scans cost `N - P` operations total, the block-wide scan of
`P` sums costs `P·log₂P`, and the final add-pass costs `N - N/P` — total
`2N + P·log₂P - P - N/P` operations. When `P` is close to `N` this
reduces to the same `O(N·log₂N)` as uncoarsened Kogge-Stone; when `P` is
much smaller than `N`, it approaches `O(N)` — i.e. coarsening recovers
near-sequential work efficiency specifically when execution resources
are the binding constraint, which is also exactly when it matters most.
Coarsening also reduces control divergence/barrier overhead overall
(more work shifted into divergence-free sequential per-thread code), and
— set up for §11.9 — lets the programmer independently choose whether
coarsening shrinks the thread count per fixed segment size, or grows the
segment size per fixed thread count (the latter also shrinks the number
of *blocks*, which matters once consolidating multiple blocks' segments
is in play).

## 11.7 Register tiling to avoid shared memory access latency (p. 270)

Since each thread's `COARSE_FACTOR`-sized subsegment is private to that
thread, keeping it in shared memory (as Fig. 11.12 does) is wasteful —
it can live in a small per-thread **register** array instead (Fig.
11.13's `buffer_r`, `#pragma unroll`-annotated so the compiler reliably
keeps its constant-indexed accesses in registers rather than spilling to
local memory). Shared memory is still used, but now only as a
**coalescing-enabling intermediary**: data is staged through it purely
to turn global-memory loads/stores coalesced (the same role shared
memory played in Chapter 6's corner-turning optimization), not for data
reuse per se. Two further checklist (Chapter 6) optimizations are named
but left as an exercise: vector loads/stores for the chunked
global-memory transfers, and padding to eliminate the strided
shared-memory bank-conflict pattern that still exists when staging
through `buffer_s`.

## 11.8 Memory bandwidth considerations (p. 272)

A scan of `N` floats loads `N` and stores `N` floats (`4N` bytes in,
`4N` bytes out) while doing `N-1` additions — an arithmetic intensity
of `≈⅛ FLOP/B`, far below an H100's 20.0 FLOP/B compute-bound
threshold, making scan **deeply memory-bound**. On an H100 (3.35 TB/s),
the absolute best possible scan throughput is `3.35e12/8 ≈ 419×10⁹`
elements/sec — a hard ceiling no scan kernel can exceed, since it
assumes every byte is touched exactly once and all non-memory work is
perfectly hidden behind memory latency (unrealistic even for a trivial
copy kernel; **>80% bandwidth utilization** is the chapter's practical
bar for "well-optimized"). Fig. 11.13's register-tiled, fully-coalesced,
no-redundant-access kernel is positioned as meeting this bar — but only
for a *single block's local segment* scan; consolidating segments across
blocks (next section) risks reintroducing extra global memory traffic.

## 11.9 Consolidating block segments for a global scan (p. 273)

Scaling scan beyond one block's segment needs an **inter-block** scan to
fold each block's running total into all *later* blocks' results — this
section's central challenge.

**Scan-scan-add across blocks** (Fig. 11.14): each block locally scans
its segment, contributes its segment's total to a global inter-block
scan, then adds the appropriate preceding-blocks total to its own
segment. Implemented naively as **three separate kernel launches** (so
the inter-block synchronization is just "the next kernel doesn't start
until the previous one's grid finishes"), this costs `(16 + 8/S)·N`
bytes of global traffic for segment size `S` — for large `S`, ≈16 B/elem
(vs. the 8 B/elem ideal), i.e. **half** the ideal 419×10⁹ elements/sec
peak, because the full `N`-element intermediate result must be written
out after kernel 1 and read back in by kernel 3.

**Reduce-scan-scan decomposition** (Fig. 11.15) trims this: stage 1 is
a cheap **reduction** (not a full scan) per segment — writing only one
total per block, not the whole scanned segment — stage 2 scans those
totals, stage 3 does the actual local scan-and-add in one pass. This
needs only `≈12` B/elem for large `S` (reduction kernel from Chapter 10
avoids storing `N` elements), reaching **2/3 of ideal peak**
(279×10⁹ elements/sec) — better than scan-scan-add's three-kernel
approach, though the choice between decompositions is not universal:
reduce-scan-scan wins when minimizing *global memory* accesses matters
most (true across blocks), while scan-scan-add wins when minimizing
*synchronization* matters most (true across warps/threads within a
block, where data already sits in registers and sync is comparatively
expensive) — the two sections's choices (§11.4/11.6 vs. §11.9) aren't
inconsistent, they're each locally optimal for their own bottleneck.

Both multi-kernel approaches still load the full input **twice**
(stage 1 and stage 3) — falling short of the true ideal of reading input
and writing output exactly once each. Reaching that needs **single-
kernel** inter-block synchronization (no kernel-launch boundary),
explored via three alternatives (Fig. 11.16): (a) **grid-wide barrier**
(all blocks share sums with block 0, scan there, redistribute) — needs
all blocks co-resident (cooperative groups, Chapter 18), limiting
scannable array size, and pays two expensive grid-wide barriers; (b)
**single lookback** — each block waits only on its immediate
predecessor's scanned sum (a **unidirectional synchronization**: later
blocks wait on earlier ones, not mutual, so blocks needn't be
co-resident, only scheduled in order) — but creates an `N/S`-long
dependency chain, a long critical path; (c) **decoupled lookback** —
each block looks back as far as needed until it finds an already-
*scanned* (not just reduced) predecessor, trading some redundant
recomputation for a much shorter critical path (an extreme case of
**multiple lookback**, which can also look back a fixed bounded
distance for a tunable latency/redundancy tradeoff).

Fig. 11.17's `interBlockScan` implements single lookback with CUDA
atomics: a `flags` array (device-scope, `cuda::atomic_ref`) signals
when each block's scanned sum is ready; a block's leader thread spins
(`fetch_add(0, memory_order_acquire)`) until the predecessor's flag is
set, reads the predecessor's scanned sum, adds its own, writes its own
scanned sum, then sets its own flag via `fetch_add(1,
memory_order_release)`. The `acquire`/`release` memory orders (as
opposed to §9's `relaxed`) are essential here, not just a performance
knob: `acquire` prevents the subsequent sum-read from being reordered
*before* the flag check, and `release` prevents the preceding sum-write
from being reordered *after* the flag-set — without this, the whole
handshake could be silently broken by reordering. Fig. 11.18 integrates
this into a complete single-kernel global scan: a **dynamically
assigned** block index (via an atomic counter incremented by one thread
per block, Fig. 11.18 lines 04-10) is required — without it, the
hardware's default `blockIdx.x` scheduling order wouldn't guarantee
earlier-indexed blocks actually *start executing* before later ones,
which unidirectional synchronization depends on to avoid deadlock.

## 11.10 Parallel scan with the Brent-Kung algorithm (p. 282)

Presented for completeness, as an alternative to Kogge-Stone with
strictly *better* theoretical work efficiency at the cost of more
sequential steps. Its two-phase structure (Fig. 11.19, 16-element
example): a **reduction tree** phase builds partial sums bottom-up using
the minimal `N-1` additions (updating positions at indices `2n-1`, then
`4n-1`, then `8n-1`, ... — `N/2 + N/4 + ... + 1 = N-1` total ops,
matching Chapter 10's reduction exactly), followed by a **reverse tree**
phase that redistributes those partial sums back out to complete every
other position's final value (Fig. 11.20 traces, for each array
position, exactly how many "black/gray/white" partial-sum
contributions it still needs after the reduction phase, showing no
position ever needs more than `log₂N - 1` contributions, each a power-
of-two distance away) — costing `N - 1 - log₂N` more additions. Total:
`2N - 2 - log₂N = O(N)` operations, matching sequential work efficiency,
but needing `2log₂N` sequential *steps* (vs. Kogge-Stone's `log₂N`) —
genuinely more total *steps*, in exchange for genuinely less total
*work*.

In practice, the book notes the Kogge-Stone algorithm still wins once
thread coarsening is applied (§11.6): coarsening already shifts most of
the actual work onto work-efficient *sequential* per-thread scans,
leaving only a small, already-`O(log P)`-sized parallel scan at the warp
level — at that scale, Brent-Kung's theoretical work savings barely
matter, while its extra steps are a straightforward cost; and since SIMD
hardware consumes a full warp's resources regardless of how many lanes
are active, "wasted" Kogge-Stone work in inactive lanes is effectively
free, whereas Brent-Kung's reduced work doesn't reduce resource
consumption the same way. Implementing Brent-Kung as a working CUDA
kernel is left entirely as an exercise.

## 11.11 Summary (p. 285)

Scan converts seemingly-sequential recurrence-based computation into
genuine parallel work, and is an important resource-allocation primitive
in its own right. The simple Kogge-Stone algorithm is fast and
conceptually simple but not work-efficient (`O(N·log₂N)` vs. sequential
`O(N)`) — a real cost once hardware resources are the binding
constraint, since the execution-unit count needed to merely *break
even* with sequential grows with dataset size. Synchronization overhead
was cut via double-buffering and warp-level primitives; work efficiency
was recovered via thread coarsening (work-efficient sequential
per-thread scans feeding a smaller, still-Kogge-Stone-style parallel
stage). Scaling beyond one block needs an inter-block consolidation
step, for which scan-scan-add and reduce-scan-scan were compared
(reduce-scan-scan costs fewer bytes, but a true single-pass kernel
needs unidirectional synchronization — single or decoupled lookback —
with careful dynamic block-index assignment to avoid deadlock). The
Brent-Kung algorithm trades more steps for genuinely less total work,
though that advantage is largely moot once coarsening is already
applied. As with reduction, production code should generally prefer an
already-optimized library (Thrust, CUB) over a hand-rolled scan.

## 11.12 Exercises (p. 286)

Problems include: hand-tracing a Kogge-Stone inclusive scan step by step
on a given 8-element array (Q1); proving that control divergence in the
Kogge-Stone kernel (Fig. 11.3) is confined to the first warp for
stride values up to half the warp size (Q2); computing the total add
operations for a 2048-element Kogge-Stone scan (Q3); modifying the
register-tiled kernel (Fig. 11.13) to add vector loads/stores and
eliminate shared-memory bank conflicts (Q4); modifying the
`interBlockScan` device function (Fig. 11.17) to use decoupled lookback
instead of single lookback (Q5); hand-tracing the same 8-element array
under the Brent-Kung algorithm (Q6); and implementing a complete
block-level Brent-Kung CUDA kernel (Q7). Not implemented in this repo's
samples (end-of-chapter exercises are out of scope — see the repo root
README).
