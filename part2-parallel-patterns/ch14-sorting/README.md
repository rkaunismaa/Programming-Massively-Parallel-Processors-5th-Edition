# Chapter 14: Sorting

Source: *Programming Massively Parallel Processors*, 5th ed., Ch. 14 (pp. 329-347).

| File | Book section | Technique |
|------|-------------|-----------|
| `01_odd_even_sort.cu` | §14.2 | Parallel odd-even transposition sort (Figs. 14.1-14.2) |
| `02_merge_sort.cu` | §14.3 | Parallel merge sort: independent segment sorts + iterative co-rank-based merge doubling (Fig. 14.3) |
| `03_radix_sort_basic.cu` | §14.5 | Parallel radix sort iteration, one thread per key, scan-based destination index (Fig. 14.7) |
| `04_radix_sort_coalesced.cu` | §14.6 | Radix sort iteration optimized for coalesced global memory access via a shared-memory local sort (Figs. 14.8-14.9) |
| `05_radix_sort_coarsened.cu` | §14.8 | Thread coarsening applied to the coalesced radix sort iteration (Fig. 14.13) |

All five files sort arrays of integers, check the GPU result against a CPU
`std::sort` reference for **exact equality** (a valid sort of a given
multiset of values is unique, so exact element-wise comparison is correct
even for a non-stable CPU reference), and print a timing line plus
`PASS`/`FAIL`.

**A note on scope.** §14.3, §14.6, and §14.8 each end their prose with some
version of "we leave the implementation ... as an exercise for the
reader" -- but each section first walks through the technique in detail,
with dedicated figures (14.3, 14.8, 14.9, 14.13) illustrating the exact
mechanism. Per this project's task brief, files 02/04/05 implement those
described (not merely named) techniques, grounded in that walkthrough text
and figures -- consistent with how earlier chapters in this project (e.g.
`ch13-merge/04_merge_coarsened.cu`, for §13.8's thread-coarsening
discussion, which carries the identical "left as an exercise" framing) have
already implemented main-body optimizations that don't happen to come with
a printed code listing. The numbered `§14.11 Exercises` list itself (multi-
bit radix, etc.) is not implemented here; §14.7's multi-bit radix
generalization is likewise out of scope -- every file in this chapter uses
a 1-bit radix, the only radix the chapter's body text gives worked figures
for.

## §14.2 Parallel odd-even sort -- `01_odd_even_sort.cu`

Odd-even transposition sort alternates between two disjoint sets of
adjacent-pair comparisons -- an EVEN step compares/swaps (0,1), (2,3), ...
and an ODD step compares/swaps (1,2), (3,4), ... -- so each step can run as
one race-free kernel launch (no thread's pair overlaps another's). The
host launches one kernel per step, alternating even/odd, until a full
even+odd round makes no swaps, capped at `n` steps -- the bound §14.2
gives for guaranteed convergence.

§14.2's text (walking through Fig. 14.2) explicitly discusses the
`hasChanged` flag: many threads writing `1` to it is technically a data
race, but a *benign* one since every writer stores the same value
(idempotence) -- while also noting this still violates the C++ memory
model in principle and that "conservative programmers should use an atomic
operation." This file follows that advice and uses `atomicExch`.

Test sizes include `n=1`, `n=2`, and an odd `n=2047` to exercise the
boundary check on the last (possibly incomplete) pair. Timing includes the
full convergence loop (many small kernel launches plus the per-step
`hasChanged` readback), not a single launch -- reflective of how this
O(N)-iteration algorithm actually runs.

## §14.3 Parallel merge sort -- `02_merge_sort.cu`

Implements Fig. 14.3's structure: divide the input into many small
segments, sort each segment independently and in parallel (one thread per
segment, sequential insertion sort), then repeatedly merge every pair of
adjacent same-length sorted segments into a double-length sorted segment
until one segment (segLen >= n) remains -- O(log(n/segLen0)) stages,
matching §14.3's stated O(log N) iteration count.

Per this chapter's task brief, the merge step is built from a co-rank
function and a co-rank-based merge kernel implemented **locally in this
file** (§14.3: "We have already seen how to parallelize a merge operation
in Chapter 13") rather than including `ch13-merge`'s files -- every chapter
in this project is self-contained. `co_rank()` is the same binary search as
`ch13-merge/01_merge_basic_parallel.cu`'s; `mergeStageKernel` assigns one
thread per **output element** (not a coarsened chunk, unlike Ch. 13's
kernels): a thread's global index picks out which segment pair it belongs
to and its local output rank `k` within that pair, `co_rank(k)` gives the
exact split point in the two input segments, and the thread writes
whichever of the two candidate elements is next in stable merge order at
that split.

To keep every stage's segment pairing exact (no partial last segment to
special-case), test cases use `n` that is an exact power-of-two multiple of
the initial segment length -- including `segLen0=1` (no initial-sort work
at all; the array is sorted purely by `log2(n)` rounds of merge-doubling)
and the degenerate `n=1` case.

## §14.5 Parallel radix sort, one thread per key -- `03_radix_sort_basic.cu`

Implements Fig. 14.7 directly: each thread loads its key, extracts the
current iteration's bit (`bit = (key >> iter) & 1`), and the block
collaborates on an **exclusive scan** of the bits array -- since bits are
0/1, the scan gives "# ones before" each position. Each thread then derives
its key's destination exactly as §14.5 derives it from first principles:

```
bit == 0:  destination = i - (#ones before i)
bit == 1:  destination = n - (#ones total) + (#ones before i)
```

The host loops this kernel once per bit (`NUM_BITS=16`, since test keys are
drawn from `[0, 2^16)`), matching §14.5's point that iterations are
sequential (each depends on the previous iteration's full output) while
only the work *within* an iteration is parallel.

This file implements the scan as a single **block-wide** Hillis-Steele
scan (`n <= 1024`, one block) -- exactly a grid-wide scan when the grid is
one block -- to keep the file focused on Fig. 14.7's destination-index
derivation without pulling in Chapter 11's multi-block scan machinery.
Files 04 and 05 extend this to many blocks.

## §14.6 Optimizing for memory coalescing -- `04_radix_sort_coalesced.cu`

File 03's kernel writes each key directly to its data-dependent
destination -- adjacent threads generally do *not* write to adjacent
addresses, so those writes can't be coalesced. §14.6 fixes this with the
"stage through shared memory" strategy (one of the three general
coalescing techniques from Chapter 6): each block first sorts its own tile
locally in shared memory (a local 1-bit radix partition -- zero-bucket keys
before one-bucket keys, computed exactly like file 03's kernel but
block-scoped), then writes that reordered tile back to a `staged` global
array at the *same* tile offset it read from, which is fully coalesced
since the scrambling happened only in shared memory.

Two kernels plus a small host step implement Figs. 14.8-14.9:

1. `localSortAndCountKernel` (Fig. 14.8): local sort + coalesced
   stage-back, and records each block's local bucket sizes (# zeros, #
   ones) into a `counts` table laid out exactly as Fig. 14.9 shows:
   row-major, all blocks' zero counts followed by all blocks' one counts.
2. An exclusive scan over that `counts` table (only `2*numBlocks`
   elements) gives each block's global starting offset for its zero bucket
   and its one bucket (Fig. 14.9). Since this table is tiny relative to
   `n`, it's scanned on the host rather than pulling in Chapter 11's
   grid-wide scan kernel (`ch11-scan/06_scan_global_multi_block.cu`) --
   keeping this file focused on the coalescing technique itself.
3. `scatterToGlobalKernel`: each block reads back its staged, locally-sorted
   tile and writes the zero-bucket keys starting at its global zero offset
   and the one-bucket keys starting at its global one offset -- within each
   sub-range, consecutive threads write consecutive addresses, exactly the
   pattern Fig. 14.8's text describes.

Test cases include `n=1000` and `n=70000` specifically to exercise a
partial last block (`validCount < BLOCK_DIM`), since padding threads must
neither read out-of-bounds nor be miscounted into either bucket.

## §14.8 Thread coarsening to improve coalescing -- `05_radix_sort_coarsened.cu`

File 04's tile size equals `BLOCK_DIM`; more (smaller) blocks for the same
`n` means smaller local buckets (less to coalesce when written out) and a
bigger `counts`/scan table (`2*numBlocks` rows). §14.8's fix is thread
coarsening: each thread now owns `COARSE_FACTOR` keys instead of 1, so a
block's tile grows to `BLOCK_DIM*COARSE_FACTOR` keys without adding
threads -- the same `n` needs `COARSE_FACTOR`x fewer blocks, giving bigger
local buckets and a `COARSE_FACTOR`x smaller scan table, matching §14.8's
stated benefit directly.

The local partition now needs a block-wide scan over `TILE_SIZE =
BLOCK_DIM*COARSE_FACTOR` bits using only `BLOCK_DIM` threads: each thread
first does a sequential inclusive scan of its own contiguous
`COARSE_FACTOR`-element chunk, the block-wide Hillis-Steele scan then runs
over just the `BLOCK_DIM` per-thread chunk totals, and an add-back pass
folds each thread's exclusive total into its chunk -- the same
coarsened-scan structure as `ch11-scan`'s register-tiled kernels,
reimplemented locally here. Global loads/stores use the "stride by
`BLOCK_DIM`" indexing pattern (tile position `p = c*BLOCK_DIM+tid`), which
is simultaneously coalesced (fixed `c`, consecutive `tid` -> consecutive
address) *and* order-preserving (`p` maps 1:1, in increasing order, to
`blockStart+p`) -- the same load pattern `ch11-scan`'s coarsened kernels
use, applied here to keep the local radix partition's stability intact.

`BLOCK_DIM=128`, `COARSE_FACTOR=8` (`TILE_SIZE=1024`): for `n=65536`, file
04 uses 256 blocks of 256 keys each; this file uses 64 blocks of 1024 keys
each -- a 4x-smaller `counts`/offsets scan table and 4x-larger local
buckets, for the identical sort result.

## Results

```
== bin/01_odd_even_sort ==
Parallel odd-even sort (§14.2, Fig. 14.2):
n=1: 0.0209 ms  [match]
n=2: 0.0351 ms  [match]
n=1000: 18.6799 ms  [match]
n=2047: 38.0425 ms  [match]
n=4096: 79.8200 ms  [match]
PASS
== bin/02_merge_sort ==
Parallel merge sort (§14.3, Fig. 14.3):
n=256 segLen0=16 (segments=16, merge stages=4): 0.1677 ms  [match]
n=65536 segLen0=64 (segments=1024, merge stages=10): 2.0012 ms  [match]
n=4096 segLen0=1 (segments=4096, merge stages=12): 0.2346 ms  [match]
n=1 segLen0=1 (segments=1, merge stages=0): 0.0060 ms  [match]
PASS
== bin/03_radix_sort_basic ==
Parallel radix sort, one thread per key (§14.5, Fig. 14.7):
n=1 (16 bits, single block): 0.1966 ms  [match]
n=777 (16 bits, single block): 0.1984 ms  [match]
n=1000 (16 bits, single block): 0.1987 ms  [match]
n=1024 (16 bits, single block): 0.1980 ms  [match]
PASS
== bin/04_radix_sort_coalesced ==
Radix sort optimized for memory coalescing (§14.6, Figs. 14.8-14.9):
n=1 (16 bits, 1 blocks of 256): 0.4681 ms  [match]
n=1000 (16 bits, 4 blocks of 256): 0.4714 ms  [match]
n=65536 (16 bits, 256 blocks of 256): 0.6525 ms  [match]
n=70000 (16 bits, 274 blocks of 256): 0.6578 ms  [match]
PASS
== bin/05_radix_sort_coarsened ==
Thread-coarsened radix sort (§14.8, Fig. 14.13):
n=1 (16 bits, 1 blocks, tile=1024, coarse=8): 0.9008 ms  [match]
n=1000 (16 bits, 1 blocks, tile=1024, coarse=8): 0.8627 ms  [match]
n=65536 (16 bits, 64 blocks, tile=1024, coarse=8): 0.8844 ms  [match]
n=70000 (16 bits, 69 blocks, tile=1024, coarse=8): 0.8916 ms  [match]
PASS
```

All five binaries also pass under `compute-sanitizer --tool memcheck` (0
errors) at `-arch=sm_75`.

Build and run all samples in this chapter:

```sh
make run
```
