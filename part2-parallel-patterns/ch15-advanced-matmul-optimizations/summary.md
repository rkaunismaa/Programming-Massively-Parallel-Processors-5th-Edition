# Chapter 15 — Advanced Optimizations for Matrix Multiplication

Section-by-section summary of the book content. Page numbers are printed
book pages (Chapter 15 runs pp. 349–370).

## 15.1 Background (p. 349)

Matrix multiplication (matrix-matrix multiplication) generalizes
vector-matrix multiplication: multiplying a vector `v` by matrix `B`
(`u = v×B`) linearly transforms `v`, where column `j` of `B` gives the
coefficients combining all of `v`'s elements into `u`'s element `j`
(e.g. a 3D coordinate rotation/scale in graphics). Stacking multiple
vectors as the rows of a matrix `A` and computing `D = A×B` applies
that same transform to every row at once. Adding a scale factor and an
additive (scaled) matrix gives the general form `D = α·A×B + β·C`,
known as **GEMM** (General Matrix Multiply) — since the `A×B` step is
GEMM's most compute-intensive part, the chapter focuses there. Matrices
may be stored row-major or column-major (possibly as a result of
transposing, Chapter 6); the chapter assumes both inputs are row-major,
noting the same techniques generalize to other combinations (as
production GEMM libraries support). Historically, matmul in most
applications involved modest matrix dimensions (linear transforms on
small vectors); deep learning's convolutional layers (Chapter 19) and
attention layers (Chapter 20) introduced genuinely **large-scale**
matrix multiplications, which is specifically what motivates this
chapter's optimizations — they're less profitable at small scale.

## 15.2 Data reuse analysis (p. 350)

Reviews Chapter 5's tiled matmul kernel with **general (non-square)**
tile dimensions: a thread block computes an `m×n` output tile, using
`m×k` and `k×n` input tiles (Fig. 15.1). Per pair of input tiles, the
block does `2·m·n·k` FLOP (k multiplies + k adds per output element,
`m·n` of them) while loading `4·k·(m+n)` bytes — giving arithmetic
intensity `0.5·m·n/(m+n)` FLOP/B, notably **independent of `k`**.
Growing `m` and `n` (larger output tiles) directly raises this ratio:
`m=n=32` gives 8 FLOP/B (memory-bound on an H100); `m=n=128` gives 32
FLOP/B (compute-bound on an H100). Intuition: a larger output tile lets
the *same* loaded input tile be reused across more output elements —
concretely, quadrupling `n` from 32 to 128 means one thread block (one
input-A-tile load) now computes what used to take four separate blocks
(four redundant loads of the same tile), eliminating that redundancy.

## 15.3 Using larger tiles with thread coarsening (p. 352)

Using larger output tiles means each thread must compute **multiple**
output elements and load multiple input elements — a thread-coarsening
optimization, directly analogous to Chapter 8's stencil coarsening. The
book introduces a two-level tile hierarchy (Fig. 15.2): a **block-level
output tile** (`bM×bN`, computed by one thread block) further divided
into **thread-level output tiles** (`tM×tN`, one per thread) — e.g. a
128×128 block-level tile with a 16×16 thread grid gives each thread an
8×8 (`tM×tN`) thread-level tile. (The same hierarchy idea extends to
other levels too — warp-level tiles are used in §15.5 — but §15.3–15.4
focus only on block- and thread-level tiles.)

Fig. 15.3's `mm_tiled_kernel`: each thread computes its block-level
indices (`bRow`/`bCol`), then its thread-level indices within the block
(`tRow`/`tCol`), declares a local register array `C_r[tM][tN]`
(zero-initialized via the `clear()` device function, Fig. 15.4 — fully
`#pragma unroll`-ed loops so the compiler can place it in registers,
following Chapter 5/6's constant-index-array rule), then iterates over
input-tile pairs along `k` in chunks of `bK`: `loadTile()` (Fig. 15.5)
cooperatively loads each `bM×bK`/`bK×bN` pair into shared memory
`A_s`/`B_s` (handling the case where thread count < tile element count
by looping over sub-tiles, with a bounds check per element), a
`__syncthreads()`, then `mm()` (Fig. 15.6) has each thread compute its
thread-level tile's contribution from its corresponding `tM×bK` row-
sub-tile of `A_s` and `bK×tN` column-sub-tile of `B_s` (fully unrolled
triple-nested loop, keeping `C_r` in registers), another
`__syncthreads()` before the next iteration overwrites the shared
tiles. After all `k`-chunks, `writeTile()` (Fig. 15.7) writes each
thread's `C_r` to global memory with a bounds check per element
(handling non-tile-multiple matrix dimensions). Vector loads/stores
(Chapter 6) for `loadTile`/`writeTile` are both named as further
optimizations, left as exercises.

## 15.4 Register tiling of the input tiles (p. 357)

Even with shared-memory tiling, each thread still **re-reads the same
shared-memory element from scratch on every use** (once per output
element it contributes to) — the shared-memory *access instruction*
itself, while much faster than global memory, is still real
instruction/latency overhead compared to a register read. **Register
tiling of the input tiles**: since a thread uses the same input value
for multiple output elements, load each needed shared-memory element
into a register exactly **once**, then reuse that register value for
all the output elements it contributes to (Fig. 15.8) — processing the
thread-level input tiles as a sequence of **strips** (one `k`-value's
worth of `A`/`B` data at a time), loading each strip's elements into
small register arrays (`a_r[tM]`, `b_r[tN]`), computing that strip's
contribution to *every* output-tile element, then moving to the next
strip.

Fig. 15.9's revised `mm()`: the loop over the input tiles' inner
dimension (`i < k`, the strip index) becomes the **outermost** loop
(a loop interchange versus Fig. 15.6); each iteration loads one `A`
strip (`a_r`) and one `B` strip (`b_r`) from shared memory into
registers (both loops `#pragma unroll`-ed so `a_r`/`b_r` stay
register-resident), then a fully-unrolled inner loop nest accumulates
each strip's outer-product contribution into every `C_r[row][col]`.
This is a second, independent instance of **hierarchical tiling** —
properly integrating block-level shared-memory tiles with thread-level
register tiles — a technique that, for very large problems, extends
further still to tiling across multiple GPUs or compute-cluster nodes.

## 15.5 Coalesced storing of the output tile (p. 360)

A large (e.g. 8×8) thread-level output tile is problematic to **store**:
adjacent threads in a warp store output elements 8 apart (uncoalesced),
and even vector stores (4 consecutive elements at once) only cover half
the tile's width, so adjacent threads' vector stores still land 4
elements apart — still not coalesced. Fix: **rearrange which output
elements a thread is responsible for**, so that threads in the same
warp own *physically adjacent* output elements small enough (4×4) that
each thread's row fits in one 4-element vector store, and adjacent
threads' vector stores land exactly adjacent to each other (fully
coalesced).

Fig. 15.10 works this out via a three-level hierarchy: the block-level
output tile is partitioned into **warp-level** tiles (one per warp in
the block — e.g. a 128×128 block tile over 8 warps, arranged 2×4, gives
each warp a 64×32 sub-tile); each warp-level tile is further split into
**4 quadrants**, with each of the warp's 32 threads (arranged 8×4)
taking one 4×4 sub-tile from *each* quadrant — so a thread's logical
8×8 output tile is now physically stored as **four separate, smaller
4×4 tiles**, each small enough for one coalesced vector-store
instruction, and positioned so that adjacent threads' corresponding 4×4
sub-tiles are physically adjacent in memory (Fig. 15.10's right side).
`mm()` and `writeTile()` need revising to iterate over these four
sub-tiles instead of one monolithic tile; left as an exercise.

## 15.6 Eliminating bank conflicts (p. 361)

Large per-thread output tiles also mean threads access **shared
memory** in a strided pattern — and the running 8×4-thread-per-warp
configuration (from §15.5) causes **8-way bank conflicts** when loading
`A_s` strips: with an 8-wide block-level `A` tile, the first elements
loaded by consecutive 4-thread groups (threads 0-3, 4-7, 8-11, ...) all
have linear indices that are multiples of 32 (0, 32, 64, 96, ...) — all
landing in bank 0 (Fig. 15.11a traces all 8 groups hitting the same
bank). Fixed with **padding** (Chapter 6): adding one padding column to
`A_s`'s declaration (`bM × (bK+1)` instead of `bM × bK`) changes the
linear-index stride from 32 (= 0 mod 32, causing conflicts) to 33 (co-
prime with 32), scattering the same accesses across distinct banks
(Fig. 15.11b verifies banks 0, 4, 8, ... for the sampled elements — no
conflicts). The code change is minimal: redeclare `A_s` with the extra
column and update the leading dimension passed to `loadTile()`/`mm()`
from `bK` to `bK+1`. `B_s` needs no such fix: its strip is 32
*consecutive* shared-memory elements per warp, naturally spanning all
32 distinct banks already.

## 15.7 Occupancy considerations (p. 363)

Large tiles increase arithmetic intensity (good) but pressure two SM
resources that bound occupancy: **shared memory** and the **register
file**. Shared-memory pressure is easy to relieve: since intensity
(§15.2) is independent of `k`, a smaller `bK` (e.g. 8, instead of a
value tied to `m`/`n`) shrinks `A_s`/`B_s` to a modest total (e.g. 4 KB
each at `m=n=128`) without hurting arithmetic intensity at all.
**Register pressure is the harder problem**: an 8×8 thread-level output
tile alone needs 64 registers per thread; adding register-tiled input
strips (§15.4) adds ~8 more, before even counting loop-unrolling-driven
temporaries (loop unrolling, while good for instruction scheduling,
directly *increases* register demand by multiplying live temporary
values). In practice such kernels hit the **hard per-thread register
cap** (255 registers/thread on a modern GPU), which — given a typical
64k-register SM register file — caps the SM to just **256 resident
threads**, only **12.5% occupancy** (256 of 2048) — directly explaining
why the chapter's running example uses a 256-thread block size (§15.3).

## 15.8 Software pipelining (p. 364)

Despite the low occupancy, register tiling's arithmetic-intensity
benefit is judged worth it — but low occupancy brings two real
drawbacks, addressed in turn. **Drawback 1** (too few memory-access
instructions in flight to saturate bandwidth): addressed by vector
loads/stores (§15.3's named-but-deferred optimization) — fewer,
larger instructions move the same data. **Drawback 2** (too few
resident threads to hide long-latency instructions via warp
switching): addressed by **aggressive loop unrolling** (already applied
throughout §15.3–15.4), which exposes independent fused-multiply-add
instructions the compiler can schedule between long-latency
instructions and their consumers, so cores rarely stall even without
many other warps to switch to.

The **remaining** long-latency concern is specifically memory loads and
barrier synchronization — normally hidden by having a *different*
thread block's compute phase overlap this block's memory-wait phase,
but with occupancy capped at a single resident block per SM, **no such
cross-block overlap exists**: the one resident block alternates between
a fully memory-bound phase (loading tiles, compute units idle) and a
fully compute-bound phase (computing, memory hardware idle) — Figs.
15.12(a)/15.13(a) show this explicitly, and show that the
`__syncthreads()` separating load-phase from compute-phase is what
prevents the compiler from simply interleaving the two kinds of
instructions itself (one barrier enforces a genuine read-after-write
true dependence; the other enforces only a write-after-read **false**
dependence, from reusing the same buffer).

The false dependence is eliminated with **double buffering** (Chapter 6):
separate buffers for the tile currently being computed-with versus the
tile being pre-loaded for the *next* iteration (reusing just two
buffers total — one for even iterations, one for odd — rather than one
per iteration), removing that barrier (Fig. 15.12b). This alone doesn't
let the compiler interleave the two kinds of instructions, though,
since they still live in different loop iterations — the final step is
**software pipelining**: restructure the loop so iteration `i` computes
the contribution of tile-pair `i` **while simultaneously** issuing the
loads for tile-pair `i+1`, bringing both kinds of instructions into the
same iteration where the compiler's instruction scheduler can freely
interleave them (Fig. 15.12c/15.13b). Fig. 15.14's code: pre-fetch
iteration-0's tiles before the loop; each loop iteration computes with
the *current* buffers while simultaneously pre-fetching the *next*
buffers, with the loads/computes separated only by one barrier (for the
genuine true dependence on the newly pre-fetched data) and no false-
dependence barrier (since compute and pre-fetch now target entirely
different buffer pairs — `Acurr_s`/`Bcurr_s` vs. `Anext_s`/`Bnext_s` —
with the swap happening only *after* the barrier); a final `mm()` call
after the loop handles the last tile-pair, which was pre-fetched by the
loop's final iteration. Two alternatives to compiler-driven software
pipelining are named for completeness: **warp specialization**
(dedicate some warps purely to memory access, others purely to compute,
letting the hardware's own warp scheduler interleave them) and
**specialized hardware support** for background data movement
(§15.9).

## 15.9 Specialized software and hardware support (p. 368)

Given matmul's ubiquity, NVIDIA provides highly-optimized libraries —
**cuBLAS** (standard BLAS API, including matmul), **cuDNN** (deep
learning primitives built on matmul), **CUTLASS** (open-source,
composable matmul building blocks at multiple scales), and newer
array-based interfaces like **cuTile** — such that programmers
virtually never hand-implement matmul in production; the chapter's
value is pedagogical, teaching transferable technique.

Dedicated hardware support: since Volta, NVIDIA GPUs include **tensor
cores** — special-purpose units performing a small matrix
multiplication in a single instruction, cutting instruction-
decode/dispatch overhead and achieving very high throughput via
warp-level register-exchange hardware and lower-precision arithmetic.
Programming model evolution: pre-Hopper tensor cores used **WMMA**
(Warp Matrix Multiply and Accumulate, single-warp scope); Hopper added
**WGMMA** (Warp Group Matrix Multiply Accumulate — multiple warps
cooperate on one larger matmul, can read operands straight from shared
memory reducing register pressure, and executes **asynchronously**,
reducing reliance on compiler-driven instruction scheduling); Blackwell
adds **TMEM** (Tensor Memory — dedicated on-chip storage for tensor-core
inputs/outputs, further easing register pressure). As tensor cores
raise compute throughput, data-movement rate (global↔shared memory)
can become the new bottleneck — addressed by **`LDGSTS`** instructions
(load vector data from global memory *directly* into shared memory,
skipping the register stop a vector-load-then-vector-store sequence
would otherwise need — reduces both register usage and instruction
count; emitted automatically by NVCC when it recognizes a
load-then-store pattern, or explicitly via `cuda::memcpy_async`) and, from
Hopper onward, the **Tensor Memory Accelerator (TMA)** — asynchronous
transfer of entire multidimensional tensors between global and shared
memory with no intermediate registers at all, a generalization of
`LDGSTS` from 1D vectors to full tensors, usable via `cuda::memcpy_async`
or through library routines (cuBLAS/cuDNN/CUTLASS) that use it
internally.

## 15.10 Summary (p. 370)

The chapter applied Chapter 6's general optimization checklist to
build an advanced, highly-optimized matmul kernel: larger output tiles
(raising arithmetic intensity toward compute-bound) via thread
coarsening; register tiling of input tiles to overcome the resulting
shared-memory-access-latency bottleneck; rearranged thread-to-output-
element mapping for coalesced stores; padding to eliminate shared-
memory bank conflicts. The resulting heavy register usage drives
occupancy very low, mitigated via vector loads/stores (ensuring enough
memory accesses in flight) and aggressive loop unrolling (ensuring
enough independent compute instructions in flight), with software
pipelining (or warp specialization, or dedicated async-copy hardware)
overlapping the kernel's memory-bound and compute-bound phases despite
having only one resident thread block per SM. In practice, all of this
is already embodied in production libraries (cuBLAS, cuDNN, CUTLASS)
and specialized hardware (tensor cores, TMEM, TMA) — matmul remains an
excellent vehicle for learning the underlying optimization techniques
even though real code should use the libraries.

## 15.11 Exercises (p. 370)

Three implementation exercises extending this chapter's kernel: modify
`loadTile()` (Fig. 15.5) to use vector loads instead of scalar loads
(Q1); modify `writeTile()` (Fig. 15.7) to use vector stores instead of
scalar stores (Q2); and reimplement the full tiled matmul kernel using
the §15.5 output-tile rearrangement for coalesced storing (Q3). Not
implemented in this repo's samples (end-of-chapter exercises are out of
scope — see the repo root README).
