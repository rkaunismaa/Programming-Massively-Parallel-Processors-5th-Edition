# Chapter 12: Filter

Source: *Programming Massively Parallel Processors*, 5th ed., Ch. 12 (pp. 289-302).

| File | Book section | Technique |
|------|-------------|-----------|
| `01_filter_unstable.cu` | §12.2 | Simple unstable filter (Fig. 12.2): one device-scope `cuda::atomic_ref::fetch_add` per surviving key claims its output slot |
| `02_filter_warp_aggregated_atomics.cu` | §12.3 | Coalesced ("warp-aggregated") atomics (Figs. 12.3-12.4): one atomic per warp instead of one per surviving key, via raw warp-voting intrinsics AND the cooperative-groups `coalesced_group` API |
| `03_filter_privatized.cu` | §12.4 | Privatization (Figs. 12.5-12.6): per-block private counter + output list in shared memory, merged into the public list with one atomic per block |
| `04_filter_stable.cu` | §12.5 | Simple stable filter (Figs. 12.7-12.8): order-preserving compaction via a grid-wide exclusive scan of a 0/1 "keep" flag, built on Ch. 11's single-lookback global scan |
| `05_filter_stable_coalesced_coarsened.cu` | §12.6 | Memory coalescing + thread coarsening (Figs. 12.9-12.10): per-block shared-memory gather before one contiguous global write per block, `COARSE_FACTOR=4` |
| `06_filter_in_place_stable.cu` | §12.7 | In-place stable filter (Fig. 12.11): file 04's kernel, unmodified in its ordering logic, run with input and output pointing at the same buffer |

All six files filter `unsigned int` keys under the same predicate,
`cond(val) = (val % 2 == 0)` (keep even-valued keys) -- deterministic, cheap
to check on the host, and keeps roughly half of any random input so every
test case exercises real compaction. Every file uses the same style of
deterministic synthetic input generator (`generateInput`, duplicated per
file per this repo's convention, each with its own PRNG seed): a small
linear-congruential PRNG producing `unsigned int` values in `[0, 65536)`.

**Unstable vs. stable checks.** Per §12.1's own terminology (Fig. 12.1):
files 01-03 are unstable, so their GPU output is checked as a **set**
against the CPU reference (sort-and-compare of the surviving multiset,
plus a count check) -- their surviving keys may land in any order. Files
04-06 are stable, so their GPU output is checked for **exact,
order-preserving equality** (`gpu == ref`, a plain vector comparison)
against a sequential CPU filter that preserves input order.

## §12.2 A simple parallel unstable filter -- `01_filter_unstable.cu`

Implements Fig. 12.2 directly. Every thread loads its key, and if `cond()`
holds, atomically increments a single global `outputSize` counter
(`cuda::atomic_ref<unsigned int, cuda::thread_scope_device>`,
`memory_order_relaxed`) to claim its slot, then stores its key there. §12.2
calls out the resulting bottleneck explicitly: every surviving key across
the *entire grid* contends on the *same* counter, and the hardware
serializes every one of those atomics. §12.3-12.4 (files 02-03) each attack
this contention differently.

Measured (`nvcc -arch=sm_75 -Xptxas -v`): 12 registers/thread, 0 bytes
static shared memory.

## §12.3 Coalescing atomic operations with warp-level primitives -- `02_filter_warp_aggregated_atomics.cu`

The book gives two versions of the same idea and this file implements both,
checked independently against the CPU reference:

- `filterKernelIntrinsics` (Fig. 12.3): raw warp-voting/shuffle primitives.
  `__activemask()` finds which lanes are active at the atomic site;
  `__ffs(mask)-1` picks the lowest active lane as leader;
  `__popc(mask)` gives the total slots the warp needs; the leader issues
  ONE `fetch_add`; `__shfl_sync` broadcasts the base index back to every
  active lane; `__popc(mask & ((1<<lane)-1))` (a binary prefix sum over the
  active-mask bits, citing Harris & Garland [1]) gives each thread its own
  offset within the reserved block.
- `filterKernelCoopGroups` (Fig. 12.4): the identical four steps rewritten
  against `cooperative_groups::coalesced_threads()` -- `thread_rank()==0`
  replaces `__ffs()`, `group.size()` replaces `__popc()`, `group.shfl(j,0)`
  replaces `__shfl_sync`, and `thread_rank()` itself IS the intra-warp
  offset.

Both kernels only coalesce the atomic *within one warp at one instruction*;
inter-warp ordering of `fetch_add` calls is still arbitrary, so the
compaction remains unstable, exactly like file 01. §12.3 states plainly
that "in practice, the compiler already implements the optimization of
coalescing atomic operations... so it is unnecessary for programmers to
apply it manually," and skips it entirely when only one thread in the warp
is active -- so this file makes no speedup claim over file 01; both kernels
are timed (each with its own untimed warm-up launch immediately before its
own timed launch, per this project's timing-fairness rule) purely to
demonstrate the mechanics, not to assert a winner.

Measured: 14 registers/thread for both kernels, 0 bytes static shared
memory.

## §12.4 Privatization -- `03_filter_privatized.cu`

Implements Fig. 12.6 (illustrated by Fig. 12.5) directly: `output_s[BLOCK_DIM]`
and `outputSize_s` are a per-block private output list and counter in
shared memory (thread 0 zero-initializes the counter, guarded by
`__syncthreads()`). Every thread filters into the PRIVATE list using a
`cuda::thread_scope_block` atomic -- the counter lives in shared memory and
is only ever touched by this one block's threads, so `thread_scope_block`
(cheaper than file 01/02's `thread_scope_device`) is sufficient. After a
barrier, thread 0 reserves one contiguous chunk of the PUBLIC list with a
*single* `thread_scope_device` atomic (incrementing by the block's whole
private count in one shot, not once per key), and consecutive threads copy
`output_s[threadIdx.x]` to `output[j + threadIdx.x]` -- consecutive threads
writing consecutive global addresses, which the book notes is coalesced as
a side benefit, matching file 02's coalescing side-benefit from a different
mechanism.

Measured: 9 registers/thread, 1032 bytes static shared memory
(`output_s` = `256*4` = 1024 B + `outputSize_s` = 4 B + `j` = 4 B = 1032 B,
matching the measured total exactly).

## §12.5 A simple parallel stable filter -- `04_filter_stable.cu`

Files 01-03 are unstable: whichever slot the atomic counter happens to hand
back, a key lands there, with no relation to its input position. A stable
filter instead places every surviving key at exactly the index equal to how
many *earlier* keys also survived -- the **exclusive scan** (Ch. 11) of a
0/1 "keep" flag (Fig. 12.7's worked example walks through exactly this:
`keep = [0,1,0,1,1,0,1,0,1,0,1,1,0,0,1,0]`, exclusive-scanned into each
surviving key's output index).

Fig. 12.8's kernel is written against a black-box `gridExclusiveScan(keep)`,
which the book explains is "performed within a single kernel following the
single-kernel scan implementation discussed in Chapter 11" (§11.9,
single-lookback). This file supplies that scan explicitly, reusing this
project's Chapter 11 file 06 machinery (`warpScan` / `blockScan` /
`interBlockScan`, dynamic block-index assignment via `atomicAdd` on a
global counter, device-scope `cuda::atomic_ref` lookback with
acquire/release ordering) specialized to `unsigned int` "keep" values and
converted from inclusive to exclusive (`inclusiveLocal - keep`). The kernel
body is otherwise Fig. 12.8 line-for-line, including its own boundary
convention: the thread whose global index is `N-1` writes `*outputSize`,
regardless of which block it falls in (this file adds an `i < N` bounds
check around the load/predicate that Fig. 12.8's own listing omits, since
its examples implicitly assume `N` is a multiple of the launch size).

Measured: 19 registers/thread, 40 bytes static shared memory
(`warpSums_s` = `8*4` = 32 B + `bid_s` = 4 B + `interBlockScan`'s
`previousSum` = 4 B = 40 B, matching the measured total exactly).

## §12.6 Improving memory coalescing with shared memory and thread coarsening -- `05_filter_stable_coalesced_coarsened.cu`

**Book-fidelity disclosure**: §12.6 gives no full kernel code for this
combination -- only two worked diagrams (Fig. 12.9, Fig. 12.10) -- and
explicitly states "we leave the detailed stable filter implementation with
exclusive scan, privatization, and thread coarsening as an exercise." What
this file implements is the diagrams' own described mechanics (per this
project's established convention: a mechanism the main text walks through
with a concrete example, even without accompanying source code, is in
scope; an undescribed exercise prompt is not):

- Fig. 12.9 (coalescing): file 04 writes each surviving key straight to its
  final global address, so a block's kept keys land scattered among other
  warps'/blocks' writes. Fig. 12.9's fix -- gather a block's kept keys into
  a private, compacted list in shared memory first (file 03's privatization
  idea, applied to the *stable* kernel), then write that shared buffer to
  global memory as one contiguous run with consecutive threads writing
  consecutive addresses.
- Fig. 12.10 (coarsening): give each block `COARSE_FACTOR` contiguous keys
  per thread instead of one (matching the worked example: block 0 covers
  `k0..k7`, thread 0 covers `k0-k1`, thread 1 covers `k2-k3`, ...), so each
  block's compacted shared-memory run -- and hence each contiguous global
  write -- is larger, and there are fewer of them. §12.6 also names
  coarsening's arguably bigger benefit: it coarsens the grid-wide *scan*
  itself, inherited for free here since each thread folds `COARSE_FACTOR`
  keep-bits into one local count before ever entering `blockScan`.

Built directly on file 04's `warpScan`/`blockScan`/`interBlockScan`
(unchanged); each thread now buffers up to `COARSE_FACTOR` surviving values
in a register array, gets its base offset in the block's shared compacted
list from a block-local exclusive scan of its own per-thread keep-count,
and the block writes its whole compacted run to global memory starting at
the offset the grid-wide lookback scan (same mechanism as file 04) assigns
it.

Measured: 16 registers/thread, 4140 bytes static shared memory
(`output_s` = `4*256*4` = 4096 B + `warpSums_s` = `8*4` = 32 B +
`bid_s`/`blockTotal_s` = 8 B = 4136 B measured plus 4 B of compiler
padding = 4140 B).

## §12.7 In-place stable filter -- `06_filter_in_place_stable.cu`

§12.7's claim: the single-lookback stable kernel from §12.5 already enforces
the ordering an in-place compaction needs, with **no code change** to its
scan logic. The danger is a thread overwriting a value another thread
hasn't read yet (Fig. 12.10's own worked example: `k8`'s output slot is the
same memory cell `k4` used to occupy). Two guarantees make this safe:

- **Within a block**: every thread loads its own input value into a
  register before any thread in the kernel does any writing. This file adds
  the `__syncthreads()` §12.7 calls out explicitly, right after the load,
  as a documented read/write barrier (technically redundant with
  `blockScan`'s own internal syncs, but the book frames it as a distinct,
  necessary step, so it's made an explicit, visible line here rather than
  left implicit).
- **Across blocks**: Fig. 12.11's dependence diagram shows
  `read_i -> scan_i -> scan_j -> write_j` for any `j > i`. Block `j`'s
  leader thread cannot pass `interBlockScan`'s lookback wait until block
  `i`'s leader has already published its partial sum -- which block `i`
  only does *after* block `i`'s own read step. So no block can ever write
  ahead of an earlier block's reads.

This file is file 04's kernel, unmodified in its scan/ordering logic,
called with the output pointer equal to the input pointer. Correctness is
checked by filtering into the same device buffer and comparing the
surviving prefix, in order, against the same CPU stable-filter reference
files 04-05 use.

Measured: 19 registers/thread, 40 bytes static shared memory (identical
layout to file 04).

## Results (RTX 4090, `sm_89`, binaries built for `-arch=sm_75`; this
environment's default CUDA device 0 is the 4090, not the 2070 SUPER)

```
== bin/01_filter_unstable ==
N=1024: cpu kept=512 gpu kept=512  0.0051 ms  [match]
N=100000: cpu kept=49991 gpu kept=49991  0.0051 ms  [match]
N=1048576: cpu kept=524288 gpu kept=524288  0.0215 ms  [match]
PASS
== bin/02_filter_warp_aggregated_atomics ==
N=1024: cpu kept=512 | intrinsics kept=512 0.0051 ms [match] | coop_groups kept=512 0.0041 ms [match]
N=100000: cpu kept=50009 | intrinsics kept=50009 0.0054 ms [match] | coop_groups kept=50009 0.0051 ms [match]
N=1048576: cpu kept=524288 | intrinsics kept=524288 0.0215 ms [match] | coop_groups kept=524288 0.0215 ms [match]
PASS
== bin/03_filter_privatized ==
N=1024: cpu kept=512 gpu kept=512  0.0051 ms  [match]
N=100000: cpu kept=49991 gpu kept=49991  0.0051 ms  [match]
N=1048576: cpu kept=524288 gpu kept=524288  0.0080 ms  [match]
PASS
== bin/04_filter_stable ==
N=1024 (blocks=4): cpu kept=512 gpu kept=512  0.0083 ms  [match]
N=100000 (blocks=391): cpu kept=50009 gpu kept=50009  0.1843 ms  [match]
N=1048576 (blocks=4096): cpu kept=524288 gpu kept=524288  1.9148 ms  [match]
PASS
== bin/05_filter_stable_coalesced_coarsened ==
N=1024 (blocks=1, COARSE_FACTOR=4): cpu kept=512 gpu kept=512  0.0061 ms  [match]
N=100000 (blocks=98, COARSE_FACTOR=4): cpu kept=49991 gpu kept=49991  0.0510 ms  [match]
N=1048576 (blocks=1024, COARSE_FACTOR=4): cpu kept=524288 gpu kept=524288  0.4833 ms  [match]
PASS
== bin/06_filter_in_place_stable ==
N=1024 (blocks=4): cpu kept=512 gpu kept=512 (in-place)  0.0072 ms  [match]
N=100000 (blocks=391): cpu kept=50003 gpu kept=50003 (in-place)  0.1874 ms  [match]
N=1048576 (blocks=4096): cpu kept=524288 gpu kept=524288 (in-place)  1.9077 ms  [match]
PASS
```

Files 04-06's larger `N=1048576` case (`~2 ms`, versus files 01-03's
`~0.02-0.03 ms` at the same size) is dominated by the single-lookback
scan's inter-block critical path (4096 blocks chained sequentially for
files 04/06's one-element-per-thread launch, versus only 1024 blocks for
file 05's `COARSE_FACTOR=4` launch -- consistent with file 05 running
roughly `4x` faster than file 04 at the same `N`, matching Ch. 11 §11.9's
own point that thread coarsening shortens the lookback chain). These are
genuine measurements at this problem size, not claims the book makes
directly; they aren't used here to assert general performance rules beyond
what the numbers above actually show.

Build and run all samples in this chapter:

```sh
make run
```
