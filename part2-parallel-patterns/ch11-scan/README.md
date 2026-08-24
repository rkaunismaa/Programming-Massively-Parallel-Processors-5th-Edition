# Chapter 11: Scan

Source: *Programming Massively Parallel Processors*, 5th ed., Ch. 11 (pp. 251-287).

| File | Book section | Technique |
|------|-------------|-----------|
| `01_scan_kogge_stone.cu` | §11.2 | Single-buffer Kogge-Stone kernel (Fig. 11.3): in-place per-block scan, two `__syncthreads()` per loop iteration (true + false dependence) |
| `02_scan_kogge_stone_double_buffered.cu` | §11.3 | Double-buffered Kogge-Stone kernel (Fig. 11.5): two shared-memory buffers eliminate the false dependence, one `__syncthreads()` per iteration |
| `03_scan_warp_level_primitives.cu` | §11.4 | Warp-primitive scan-scan-add decomposition (Figs. 11.8-11.10): `__shfl_up_sync`-based `warpScan`, combined block-wide via `blockScan` |
| `04_scan_coarsened.cu` | §11.6 | Thread coarsening (Fig. 11.12): each thread sequentially scans `COARSE_FACTOR` elements in shared memory before the block-wide scan of per-thread sums |
| `05_scan_register_tiled.cu` | §11.7 | Register tiling (Fig. 11.13): each thread's subsegment lives in a local register array instead of repeatedly touching shared memory |
| `06_scan_global_multi_block.cu` | §11.9 | Single-kernel global (multi-block) scan (Figs. 11.17-11.18): dynamic block-index assignment + single-lookback inter-block scan via device-scope `cuda::atomic_ref` |
| `07_scan_brent_kung.cu` | §11.10 | Brent-Kung reduction/reverse-tree scan -- **no code figure in the book** (see below); measured head-to-head against file 01's Kogge-Stone kernel |

All seven files perform an **inclusive scan** (prefix sum) of `float` values
under addition, the chapter's running operator (§11.1: "when the scan
operator is addition, the computation is sometimes also referred to as
prefix sum... we will only present parallel algorithms and implementations
for inclusive scan throughout this chapter"). Every file uses the same
deterministic synthetic input generator (`generateInput`, duplicated per
file per this repo's convention): a small linear-congruential PRNG producing
uniform `float`s in `[0, 1)`.

**Per-block vs. global scope.** Files 01-05 and 07 all implement **per-block
(per-segment) scans** -- exactly what the book's own Figs. 11.3/11.5/11.10/
11.12/11.13 do: "we will for now implement a kernel where each thread block
performs a local parallel scan on a segment of the input... Later, in
Section 11.9, we will discuss how to consolidate these scanned segments to
produce a globally scanned output array" (§11.2). Their CPU references
(`scanSegmentsCPU`) match this scope exactly: an independent sequential
inclusive scan restarted at every segment boundary, not one continuous scan
over the whole test array. Only file 06 (§11.9) performs a true grid-wide
scan and is checked against a single continuous CPU reference (`scanCPU`).

**Floating-point tolerance.** Every correctness check uses
`nearlyEqual(gpu, cpu, 1e-2f)`, matching Chapter 10's convention: floating
point addition is not strictly associative, and every kernel here reorders
the addition tree relative to the CPU's strictly-sequential reference
(warp-level trees, thread-level coarsened partial sums, inter-block lookback
order, etc.), so a wider-than-default tolerance absorbs summation-order
rounding differences rather than asserting bit-exact equality.

## §11.2 Kogge-Stone kernel -- `01_scan_kogge_stone.cu`

Implements Fig. 11.3 directly: an in-place, per-block evolving array where,
after `k` iterations, position `i` holds the sum of up to `2^k` input
elements ending at `i`. Two `__syncthreads()` per iteration are required
(and the book works through the exact race, thread 4 vs. thread 6 at
iteration 2, that a missing second barrier would cause): the first guards a
**true** dependence (wait for the previous iteration's writes), the second a
**false** dependence (wait for all reads of the old value before any thread
overwrites it). §11.5 derives the algorithm's own work complexity:
`N*log2(N) - (N-1)` additions per segment -- worse than the sequential
algorithm's `O(N)`, "between eight and nine times more work than the
sequential code" for `N=512`.

Measured (`nvcc --resource-usage -arch=sm_75`): 10 registers/thread, 0 bytes
static shared memory (dynamically-sized `buffer_s`, one segSize-length
`float` array per launch).

## §11.3 Double-buffering -- `02_scan_kogge_stone_double_buffered.cu`

Implements Fig. 11.5: two shared-memory buffers (`buffer1_s`, `buffer2_s`)
swap roles as input/output buffer every iteration, so no thread ever writes
to a location another active thread reads in the same iteration -- the false
dependence disappears along with its barrier. Values already final (not
touched by an iteration's active threads) must be explicitly copied forward
into the new output buffer, since they are no longer retained automatically
once reads and writes target different memory. Register count is identical
to file 01 (10 registers/thread) since the same amount of arithmetic runs
per thread; shared memory doubles to hold both buffers (dynamically sized,
`2 * segSize * sizeof(float)` per launch).

## §11.4 Warp-level primitives -- `03_scan_warp_level_primitives.cu`

Implements Figs. 11.8-11.10: the scan-scan-add decomposition (Fig. 11.6)
applied at warp granularity. Every warp independently scans its own
32-element slice using `warpScan` (`__shfl_up_sync`-based, register-only, no
shared memory or barriers within the loop); the last lane of each warp
writes its warp's total to a small shared array; warp 0 scans that array
with the same `warpScan` primitive; every other warp then adds its entry
(the sum of all *preceding* warps) to its own already-scanned elements.
Total block-wide `__syncthreads()` calls drop to exactly two (after warps
publish their sums, and after warp 0 finishes scanning them), versus one per
loop iteration in files 01/02.

Measured: 14 registers/thread (up from file 02's 10 -- the extra live state
is `warpScan`'s loop-local `val`/`temp` plus `blockScan`'s lane/warp index
bookkeeping), 0 bytes static shared memory (dynamically-sized `warpSums_s`,
`segSize/32` floats per launch).

## §11.6 Thread coarsening -- `04_scan_coarsened.cu`

Implements Fig. 11.12: `COARSE_FACTOR=4`, `BLOCK_DIM=256`, so each block now
covers `1024` elements with only `256` threads. The block's segment is
loaded into shared memory in `COARSE_FACTOR` **coalesced chunks** of
`BLOCK_DIM` elements each (not per-thread contiguous reads, which the book
calls out explicitly as producing uncoalesced accesses); each thread then
re-addresses that same shared buffer **contiguously**
(`threadIdx.x*COARSE_FACTOR + c`) to run a work-efficient sequential scan of
its own subsegment; the per-thread subsegment sums are combined with file
03's `blockScan`; each thread adds the scanned sum of all preceding threads
to its own subsegment; a final coalesced store mirrors the load.

Measured: 26 registers/thread, 5152 bytes static shared memory
(`buffer_s` = `COARSE_FACTOR*BLOCK_DIM*4` = 4096 B, `warpSums_s` =
`(BLOCK_DIM/32)*4` = 32 B, `scannedThreadSums_s` = `BLOCK_DIM*4` = 1024 B;
4096+32+1024 = 5152 B, matching the measured total exactly).

## §11.7 Register tiling -- `05_scan_register_tiled.cu`

Implements Fig. 11.13, built directly on file 04: the thread's
`COARSE_FACTOR`-element subsegment is loaded from shared memory into a local
array `buffer_r` **once**, scanned entirely in registers
(`#pragma unroll`-annotated per the book, "a useful reminder to the reader
that the loop is intended to be unrolled"), and written back to shared
memory once at the end -- eliminating the repeated shared-memory
read-modify-write traffic file 04 pays for that private, never-shared data.
Shared memory is still used as an intermediary purely to keep the global
`load`/`store` coalesced (§11.7 draws the explicit parallel to Chapter 6's
corner-turning optimization), not because the register-tiled data itself is
reused there.

Measured: 5152 bytes static shared memory, identical to file 04 (same
buffer layout). Registers/thread measured at **28**, slightly *more* than
file 04's 26 -- register tiling here targets shared-memory access
*latency*, not register *count*; keeping the whole `COARSE_FACTOR`-element
subsegment simultaneously live in registers (rather than round-tripping
through shared memory one element at a time) naturally increases live
register state. This project reports the measured number rather than
assuming register tiling implies fewer registers.

## §11.9 Consolidating block segments for a global scan -- `06_scan_global_multi_block.cu`

§11.9 surveys three inter-block-scan strategies and derives their global
memory traffic (`N` = total elements, `S` = elements/block segment):

- **Three-kernel scan-scan-add**: `(4 + 8/S)*N` bytes, `~16 B/element` for
  large `S` -- half the ideal peak of `419*10^9 elements/s` the book derives
  in §11.8 for an H100 (`3.35 TB/s` bandwidth / `8 B/element` minimum
  traffic), "because it performs twice the number of global memory accesses
  due to the need to store all N intermediate values."
- **Three-kernel reduce-scan-scan**: `(3 + 3/S)*N` bytes, `~12 B/element` --
  better (`2/3` of ideal peak, `279*10^9 elements/s`), by replacing the
  first kernel's per-element scan output with a per-block reduction output.
- **Single-kernel with an in-kernel inter-block scan** (Figs. 11.16-11.18):
  avoids the extra global round-trip entirely. This file implements the
  **single-lookback** variant the book gives full kernel code for
  (Fig. 11.17's `interBlockScan` device function, Fig. 11.18's kernel).

The kernel dynamically assigns each block a logical index `bid` (an
`atomicAdd` on a global counter, executed by thread 0 the instant the block
starts running) instead of using `blockIdx.x` directly -- this is what makes
**unidirectional synchronization** deadlock-free: a block with a lower `bid`
is guaranteed to have started executing no later than any block with a
higher `bid`, so no block ever ends up waiting on one that hasn't started.
Each block's leader thread (`threadIdx.x == blockDim.x-1`) spins on a
device-scope `cuda::atomic_ref<unsigned int, cuda::thread_scope_device>`
flag for the preceding block (`memory_order_acquire`, so the following read
of that block's partial sum can't be reordered ahead of the flag check),
adds that partial sum to its own block's total, publishes the result, and
sets its own flag (`memory_order_release`, so the flag update can't be
reordered ahead of the partial-sum write). `thread_scope_device` is required
-- unlike Chapter 9's atomics, this flag is read and written across
*different blocks*, not just different threads of the same block.

The block-local portion of the kernel is, per the book, "mostly the same as
the code in Fig. 11.13" -- this file reuses file 05's register-tiled
coarsened local scan verbatim, then folds the inter-block add into the
already-scanned local segment before the final coalesced store.

This is the only file in the chapter checked against a genuinely global CPU
reference (`scanCPU`, one continuous accumulator over the whole array, not
per-segment), and the only file whose test cases include an `N` that isn't
an exact multiple of the block segment size (`N=100000`, 98 blocks, last
block zero-padded) to exercise the boundary path.

Measured: 30 registers/thread, 5160 bytes static shared memory (file 05's
5152 bytes plus 8 bytes for `bid_s` and `interBlockScan`'s `previousSum`).

**Not implemented**: §11.9 also describes **decoupled lookback** (a block
looks back through multiple preceding blocks until it finds one that's
already fully scanned, trading redundant work for a shorter critical path)
as a generalization of single lookback, but explicitly "leave[s] the
implementation of a revised `interBlockScan` device function based on
decoupled lookback as an exercise" -- out of scope here, matching this
project's convention of not implementing end-of-chapter exercises.

## §11.10 Brent-Kung algorithm -- `07_scan_brent_kung.cu`

**Book-fidelity disclosure**: unlike every other section in this chapter,
§11.10 presents the Brent-Kung scan algorithm with **no accompanying code
figure** -- only two diagrams (Fig. 11.19: up-sweep reduction tree /
down-sweep reverse tree; Fig. 11.20: per-position accumulated-range table
across reverse-tree levels) and a prose walkthrough, closing with: "We leave
the implementation of the Brent-Kung algorithm for parallel scan as an
exercise for the reader." The kernel in this file is therefore **this
project's own implementation** of the described algorithm, not a
transcription of book source code. It was verified step-by-step against the
book's own worked `N=16` example before being trusted:

- Up-sweep (reduction) phase: 4 steps, updating positions of the form
  `2n-1`, `4n-1`, `8n-1`, `16n-1` -- `8+4+2+1 = 15 = N-1` operations, matching
  the book's general formula `N/2+N/4+...+1 = N-1`.
- Down-sweep (reverse tree) phase: 3 levels at stride `4`, `2`, `1` --
  `buffer[11]+=buffer[7]` (1 op), `buffer[5,9,13]+=buffer[3,7,11]` (3 ops),
  `buffer[2,4,6,8,10,12,14]+=buffer[1,3,5,7,9,11,13]` (7 ops), totaling
  `1+3+7=11`, matching the book's own arithmetic exactly: "The number of
  operations is 16/8 − 1 + 16/4 − 1 + 16/2 − 1" = `1+3+7=11 = N-1-log2(N)`.

Total operations `2N-2-log2(N)` is `O(N)` -- better work efficiency than
Kogge-Stone's `O(N*log2(N))`. But §11.10 also states the practical
counterpoint: after thread coarsening (files 04-06) most work already runs
in efficient sequential per-thread scans, and at the warp level "the
Kogge-Stone algorithm has better performance in practice... The work avoided
by the Brent-Kung algorithm at the warp level is replaced by inactive warp
lanes that still consume execution resources because of the nature of SIMD
execution." This file measures both kernels directly rather than asserting
that claim: it re-declares file 01's exact single-buffer Kogge-Stone kernel
and runs it head-to-head against Brent-Kung, each with its own untimed
warm-up launch immediately before its own timed launch (both kernels timed
in the same process, per this project's timing-fairness rule), on identical
input and identical launch geometry. The book's own operation-count formulas
(not measured) are printed alongside the measured times for reference.

Measured: Brent-Kung 12 registers/thread vs. Kogge-Stone 10 (file 01's
number, reproduced identically here since it's the same kernel code) -- 0
bytes static shared memory for both (dynamically-sized `buffer_s`).

## Results (RTX 4090, `sm_89`, binaries built for `-arch=sm_75`; this
environment's default CUDA device 0 is the 4090, not the 2070 SUPER)

```
== bin/01_scan_kogge_stone ==
segSize=64 blocks=4 N=256: last=33.310112 (cpu) / 33.310112 (gpu)  0.0051 ms  [match]
segSize=256 blocks=8 N=2048: last=124.139740 (cpu) / 124.139725 (gpu)  0.0039 ms  [match]
segSize=1024 blocks=4 N=4096: last=519.967163 (cpu) / 519.967712 (gpu)  0.0051 ms  [match]
PASS
== bin/02_scan_kogge_stone_double_buffered ==
segSize=64 blocks=4 N=256: last=32.693962 (cpu) / 32.693954 (gpu)  0.0049 ms  [match]
segSize=256 blocks=8 N=2048: last=132.712250 (cpu) / 132.712219 (gpu)  0.0048 ms  [match]
segSize=1024 blocks=4 N=4096: last=512.101501 (cpu) / 512.101501 (gpu)  0.0051 ms  [match]
PASS
== bin/03_scan_warp_level_primitives ==
segSize=64 blocks=4 N=256: last=31.077799 (cpu) / 31.077803 (gpu)  0.0051 ms  [match]
segSize=256 blocks=8 N=2048: last=123.284737 (cpu) / 123.284721 (gpu)  0.0042 ms  [match]
segSize=1024 blocks=4 N=4096: last=510.235321 (cpu) / 510.235229 (gpu)  0.0041 ms  [match]
PASS
== bin/04_scan_coarsened ==
blocks=1 N=1024 (COARSE_FACTOR=4, BLOCK_DIM=256): last=515.229431 (cpu) / 515.229553 (gpu)  0.0051 ms  [match]
blocks=4 N=4096 (COARSE_FACTOR=4, BLOCK_DIM=256): last=518.368774 (cpu) / 518.368958 (gpu)  0.0031 ms  [match]
blocks=16 N=16384 (COARSE_FACTOR=4, BLOCK_DIM=256): last=507.926575 (cpu) / 507.926544 (gpu)  0.0041 ms  [match]
PASS
== bin/05_scan_register_tiled ==
blocks=1 N=1024 (COARSE_FACTOR=4, BLOCK_DIM=256): last=495.113403 (cpu) / 495.113281 (gpu)  0.0048 ms  [match]
blocks=4 N=4096 (COARSE_FACTOR=4, BLOCK_DIM=256): last=502.502747 (cpu) / 502.502655 (gpu)  0.0031 ms  [match]
blocks=16 N=16384 (COARSE_FACTOR=4, BLOCK_DIM=256): last=511.060425 (cpu) / 511.060303 (gpu)  0.0031 ms  [match]
PASS
== bin/06_scan_global_multi_block ==
N=4096 (blocks=4): last=2057.265869 (cpu) / 2057.266846 (gpu)  0.0082 ms  [match]
N=100000 (blocks=98): last=50031.812500 (cpu) / 50031.714844 (gpu)  0.0502 ms  [match]
N=1048576 (blocks=1024): last=524760.437500 (cpu) / 524774.750000 (gpu)  0.4823 ms  [match]
PASS
== bin/07_scan_brent_kung ==
segSize=64 blocks=4 N=256:
  Brent-Kung : last=32.613190  0.0061 ms  ops/segment(book formula 2N-2-log2N)=120.0  [match]
  Kogge-Stone: last=32.613190  0.0032 ms  ops/segment(book formula N*log2N-(N-1))=321.0  [match]
segSize=256 blocks=8 N=2048:
  Brent-Kung : last=120.574692  0.0041 ms  ops/segment(book formula 2N-2-log2N)=502.0  [match]
  Kogge-Stone: last=120.574692  0.0038 ms  ops/segment(book formula N*log2N-(N-1))=1793.0  [match]
segSize=1024 blocks=4 N=4096:
  Brent-Kung : last=501.770081  0.0051 ms  ops/segment(book formula 2N-2-log2N)=2036.0  [match]
  Kogge-Stone: last=501.770081  0.0043 ms  ops/segment(book formula N*log2N-(N-1))=9217.0  [match]
PASS
```

At these input sizes every launch finishes in a handful of microseconds,
dominated by fixed per-launch overhead rather than the algorithmic
differences the chapter analyzes (file 06's `N=1048576`/1024-block case is
the one test large enough to show measurable inter-block-scan cost, at
`~0.48 ms`). The register/shared-memory counts above and the book's own
operation-count formulas (files 07's printed `ops/segment` values) are the
quantitative claims this README backs directly, per this project's
convention of not asserting a performance number without deriving it from
the book's text or measuring it on real hardware. File 07's timing columns
are genuine same-process, same-geometry, individually-warmed-up
measurements of both kernels -- they are not used here to declare a winner
beyond what was actually measured at this problem size.

Every file times its own kernel(s) with a discarded warm-up launch
immediately before the timed region (so PTX->SASS JIT cost isn't folded
into the measurement); file 07 is the one file in this chapter that times
two kernels in-process, and gives each its own independent warm-up
immediately before its own timed launch, per this project's timing-fairness
rule.

Build and run all samples in this chapter:

```sh
make run
```
