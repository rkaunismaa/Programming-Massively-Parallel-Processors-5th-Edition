# Chapter 12 — Filter (And Warp Voting)

Section-by-section summary of the book content. Page numbers are printed
book pages (Chapter 12 runs pp. 289–302).

## 12.1 Background (p. 289)

**Filtering** removes selected items from a list according to a
criterion and compacts the remaining items to eliminate the resulting
holes — e.g. garbage collection compacting a heap after some objects
are freed, so a later large allocation has a big enough contiguous
region. (Conceptually, filter is the *inverse* of merge: viewing the
deleted and kept elements as two sub-lists, a merge would recombine
them.) Filters can be **out-of-place** (compact into a new list — needs
extra memory proportional to the whole list, which can be prohibitive
for near-heap-sized data) or **in-place** (compact within the original
array — risks one thread overwriting an element another thread still
needs to read, requiring careful synchronization, covered in §12.7).
Filters are also either **unstable** (filtered items may end up in any
relative order — fine when the list needn't stay ordered) or **stable**
(filtered items keep their original relative order — needed when
extracting/deleting from an already-sorted list, Fig. 12.1). The
chapter's convention: call list elements **keys**.

## 12.2 A simple parallel unstable filter (p. 290)

One thread per input key (Fig. 12.2): load the key, test it against
`cond()`, and if kept, atomically increment a shared **output-size
counter** in global memory (`cuda::atomic_ref`, device scope) to claim
a unique output slot `j`, then write the key to `output[j]`. Because
atomics can complete in any order, keys land in the output in
whatever order their threads happen to "win" the counter race — hence
**unstable**. The clear bottleneck: every kept key contends on the
*same* global counter, fully serialized by hardware (as Chapter 9
established) — motivating two independent mitigations: **coalesced
atomic operations** (§12.3, reduces how many atomics are issued) and
**privatization** (§12.4, spreads the remaining atomics across private
per-block counters).

## 12.3 Coalescing atomic operations with warp-level primitives (p. 291)

Rather than every active thread in a warp separately atomically
incrementing the shared counter (guaranteed contention/serialization
since they target the same location), have the warp **coordinate once**
and issue a **single combined atomic operation** on behalf of the whole
warp — a *coalesced atomic operation*, directly analogous to how
loads/stores from a warp get coalesced into fewer memory transactions
(Chapter 5). Four steps, using warp-level primitives (Fig. 12.3): (1)
**identify active threads** via `__activemask()` (a 32-bit mask, bit
`i` set iff lane `i` is active); (2) **pick a leader** — the
lowest-indexed active thread, found via `__ffs()` (find-first-set) on
the mask minus 1 (since `__ffs` reserves 0 for "no bits set"); (3) the
leader **counts active threads** with `__popc()` (population count) on
the mask, and performs *one* atomic `fetch_add` of that count,
broadcasting the returned starting offset `j` back to every other
active thread via `__shfl_sync`; (4) **each thread computes its own
output position** as `j + offset`, where `offset` is the number of
*active threads preceding it in the warp* — computed via a bitwise
**binary prefix sum**: build a mask of preceding lanes (`(1 <<
laneIdx()) - 1`), AND it with the active mask, then `__popc()` the
result. A side benefit beyond fewer atomics: active threads' output
writes land at *consecutive* positions, so the stores to global memory
also become more coalesced.

The **cooperative groups** API packages all of this bookkeeping into a
**coalesced group** — `coalesced_group activeThreads =
coalesced_threads();` directly represents a warp's active threads, with
`.size()` replacing `__popc()`/`__activemask()`, `.thread_rank()`
replacing the manual bit-mask offset computation, and `.shfl(...)`
replacing `__shfl_sync` (Fig. 12.4) — a much shorter, equivalent
kernel. In practice, **the compiler already performs this coalescing
optimization automatically** (and skips it when it can prove only one
thread in the warp is active) — the chapter has the reader implement it
by hand purely as a vehicle for introducing **warp voting functions**
(`__activemask()`, plus `__all_sync()`/`__any_sync()`/`__ballot_sync()`,
named but not used here), the `__ffs()`/`__popc()` intrinsics, and
cooperative groups' coalesced groups.

## 12.4 Privatization (p. 295)

As in Chapters 6/9, **privatization** gives each thread block its own
private output list and private counter in **shared memory** (Fig.
12.5): threads filter into their block's private list/counter
(block-scope atomics, lower latency than device-scope); once the whole
input is processed, one thread atomically reserves a contiguous region
of the right size in the *public* output list/counter (a single
device-scope atomic per block, not per kept key); finally, the block's
threads **collaboratively copy** the private list to that reserved
public region. Fig. 12.6's kernel: shared `output_s[BLOCK_DIM]` and
`outputSize_s`, initialized once by thread 0 then barrier-synced;
per-thread filtering into the private list (block-scope atomic,
mirroring Fig. 12.2's logic exactly but against the private counter);
a barrier; one thread reserves space in the public counter
(device-scope atomic `fetch_add(outputSize_s, ...)`, adding the whole
block's count at once) and shares the returned offset `j` via shared
memory, barrier again; finally every thread copies
`output_s[threadIdx.x] → output[j+threadIdx.x]` for
`threadIdx.x < outputSize_s` — a bonus coalescing benefit again, since
consecutive threads write consecutive output positions.

## 12.5 A simple parallel stable filter (p. 297)

A **stable** filter must preserve each kept key's original relative
order, so a thread can't just grab an arbitrary free output slot — it
must compute the *exact* position its key belongs at, which is the
count of kept keys *before* it in the input. That count is exactly an
**exclusive scan** (Chapter 11) over a binary `keep` array (1 if
`cond()` holds, 0 otherwise) — Fig. 12.7 walks a 16-key example by
hand: `keep = [0,1,0,1,1,0,1,0,1,0,1,1,0,0,1,0]` scans (exclusive) to
`offset = [0,0,1,1,2,3,3,4,4,5,5,6,7,7,7,8]`, and each kept key is
written to `output[offset[i]]`. Fig. 12.8's kernel: compute `keep` per
thread, call a (previously-built, Chapter 11 single-kernel)
`gridExclusiveScan(keep)` to get `offset`, write kept keys to
`output[offset]`, and have the very last thread write the scan's final
value (`offset + keep` for `i == N-1`) as the output list's total size.
A further scan optimization is noted: since the scan's *input* here is
binary, the per-segment prefix sums can be computed faster with
bitwise operations (akin to §12.3's binary prefix sum technique) rather
than a general-purpose scan.

## 12.6 Improving memory coalescing with shared memory and thread coarsening (p. 298)

In Fig. 12.8, only *some* threads in a block actually write output
(those with `keep=1`), scattered among many inactive ones in the same
warp — so writes to adjacent output positions often come from
*different warps*, missing coalescing opportunities. Fix: gather each
block's filtered keys into **shared memory** first, then have a
contiguous range of threads write that shared buffer out to global
memory as one dense chunk (Fig. 12.9) — this is itself a form of
privatization, now applied purely to recover coalescing rather than to
reduce atomic contention. **Thread coarsening** compounds the benefit
(Fig. 12.10): assigning each block *more* keys than its thread count
means fewer, larger contiguous chunks get written out, exposing more
coalescing opportunity (the tradeoff: finer-grained parallelization —
smaller chunks per block — naturally limits coalescing, so there's a
real cost to maximizing thread-level parallelism here). An arguably
bigger coarsening benefit is indirect: coarsening the filter operation
also coarsens the **scan** embedded within it, and scan (Chapter 11)
benefits substantially from coarsening (better work efficiency, less
sync overhead) — so this improvement compounds with the filter's own
gains. A full implementation combining exclusive scan + privatization +
thread coarsening is left as an exercise.

## 12.7 In-place stable filter (p. 300)

Performing filter **in-place** (output overwrites the same array as
input) risks a thread overwriting a value another thread hasn't yet
read — e.g. in Fig. 12.10's example, `k1` and `k3`'s output positions
coincide with `k0`'s and... input positions used by *other* keys, so a
write-before-read bug is possible unless ordering is enforced. Within
one block, a single `__syncthreads()` between the "read all inputs"
phase and "write all outputs" phase suffices. Across blocks (the harder
case — e.g. output position 4 being written by a later block while an
earlier block might still be reading input position 4), the required
property is: **for any blocks `i` and `j` with `j > i`, block `i`'s
read must happen before block `j`'s write**. The key insight: this
exact ordering is **already guaranteed for free** by the single-pass
(unidirectional-synchronization) global scan kernel from Chapter 11 —
its dependence structure (Fig. 12.11: `Read_i → Scan_i → Scan_j → Write_j`
for `j > i`, since `Scan_j` necessarily waits on `Scan_i` via the
lookback chain) already transitively ensures `Read_i` happens before
`Write_j`. So the *same* single-kernel scan-based stable filter
implementation (§12.5/§12.6) works correctly for in-place filtering
with **no additional synchronization needed** — adapting it concretely
is left as an exercise.

## 12.8 Related patterns (p. 301)

Several patterns share the stable filter's "move data in one
directional manner" structure and inherit many of its optimizations:

- **Removing duplicate keys from a sorted list** — a special case of
  stable filter where `cond()` keeps a key only if it differs from its
  immediate predecessor.
- **Removing rows/columns from a matrix** (e.g. adjusting layout for
  memory alignment) — resembles filtering by *index* rather than
  *value*, so each thread can compute its output position **purely
  analytically** (no scan needed) — but if done in-place, still needs
  the same one-directional block-ordering synchronization as stable
  filter (just without needing to propagate an actual running partial
  sum, since positions are analytically known).
- **Adding rows/columns to a matrix** — the mirror image: positions are
  again analytically determined, but since inserted keys move
  **outward** rather than inward, an in-place version needs the
  one-directional synchronization running in the **opposite**
  direction (later blocks' reads must complete before earlier blocks'
  writes, since earlier blocks now write into the tail of what later
  blocks still need to read).

More complex in-place data-movement patterns (e.g. matrix/tensor
transposition) are pointed to in the literature rather than covered
here.

## 12.9 Summary (p. 302)

The chapter covered parallel unstable and stable filters. Unstable,
out-of-place filters rest on atomic operations, optimized by minimizing
atomic-operation count (coalesced atomics) and maximizing output-write
coalescing (privatization). Stable, out-of-place filters rest on the
scan operation and build directly on Chapter 11's single-pass scan
kernel. In practice, the Thrust and CUB libraries already provide
optimized filter implementations; having worked through these
principles, the reader is positioned to choose and apply the right
library API for a given application rather than hand-roll one.

## 12.10 Exercises (p. 302)

Four incremental implementation exercises, each building on the last:
implement stable filter with exclusive scan as three separate kernels —
condition evaluation, exclusive scan, output generation (Q1); merge
those into a single-kernel implementation built on the one-pass scan
kernel, and explain the main advantages of doing so (Q2); add
privatization to that kernel (Q3); and add thread coarsening on top of
that (Q4). Not implemented in this repo's samples (end-of-chapter
exercises are out of scope — see the repo root README) — though the
repo's own `07_filter_remove_duplicates.cu` sample implements §12.8's
named "remove duplicate keys" special case, which the book presents as
worked explanation rather than a numbered exercise.
