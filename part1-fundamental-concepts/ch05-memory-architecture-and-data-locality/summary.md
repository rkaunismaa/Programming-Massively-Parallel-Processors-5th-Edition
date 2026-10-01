# Chapter 5 — Memory Architecture and Data Locality

Section-by-section summary of the book content. Page numbers are printed
book pages (Chapter 5 runs pp. 93–121).

## 5.1 Memory bandwidth as a performance limiter (p. 93)

Hardware has two separate limits: **peak computational throughput**
(e.g. the H100's 66.9 TFLOPS single-precision) and **peak memory
bandwidth** (e.g. the H100's 3.35 TB/s global memory bandwidth). A
kernel is **compute-bound** if the cores are busy nearly all the time
and performance is capped by how fast they execute instructions; it is
**memory-bound** if the memory channels are busy nearly all the time
and the cores sit idle waiting for data. Whether a kernel is one or the
other is determined by its **compute-to-global-memory-access ratio**
(FLOP/B, also called *arithmetic intensity*), compared against the
hardware's own threshold = peak throughput ÷ peak bandwidth (20.0
FLOP/B for the H100). The **Roofline Model** plots this visually:
arithmetic intensity on the x-axis, achieved throughput on the y-axis,
with a sloped memory-bandwidth-bound line and a flat compute-bound
line; a kernel's distance below these lines shows how efficiently it
uses the hardware.

Worked examples: the Chapter 2 vector-addition kernel has a ratio of
(1 FLOP)/(12 B) = 0.083 FLOP/B — deeply memory-bound. A **speed-of-light
analysis** on it (achieving 3 TB/s out of a 3.35 TB/s peak = 90%) shows
it's already using the hardware efficiently, with the fastest possible
runtime calculable directly from the data volume and peak bandwidth.
Matrix multiplication is *ideally* compute-bound — a perfect
implementation has ratio 0.167·N FLOP/B (167 FLOP/B at N=1024, well
above the H100's 20.0 threshold) — but the Chapter 3 naive kernel
actually achieves only 0.25 FLOP/B (worse than the ideal ratio, because
each input element is re-read from global memory on every use), making
it memory-bound in practice and leaving it at roughly 1% of the
hardware's FLOPS peak (0.08% of the tensor-core peak). This motivates
reducing global memory accesses, the chapter's central technique.

## 5.2 CUDA memory types (p. 98)

Fig. 5.1 lays out CUDA's on-chip/off-chip memory spaces, each with a
different scope and lifetime (summarized in Fig. 5.4):

- **Registers** — on-chip, per-thread, fastest. Automatic scalar
  variables are placed here by default. Private to one thread; ceases
  to exist when the thread terminates. An operand in a register needs
  no extra load instruction (vs. a global-memory operand, which needs a
  separate `load` before the arithmetic instruction can use it), and
  register accesses cost roughly an order of magnitude less energy
  than global-memory accesses.
- **Local memory** — physically lives in global memory (same latency),
  but is private per-thread. Holds things that can't fit in registers:
  statically-sized automatic arrays, spilled registers, and other
  thread call-stack data. (A small, constant-indexed automatic array
  may be register-allocated by the compiler instead — see Chapter 6.)
- **Shared memory** — on-chip, scratchpad memory allocated per
  *block*; all threads in a block see the same version of a
  `__shared__` variable, making it the vehicle for intra-block data
  sharing. Slower than registers (it's still a load operation, just a
  fast on-chip one) but far faster than global memory. Starting with
  Hopper, a thread block cluster's blocks can access each other's
  shared memory as pooled **distributed shared memory**.
- **Global memory** — off-chip DRAM, R/W by both host and device,
  visible to all threads of all kernels, persists for the whole
  application; slow, and "global variables are considered bad style"
  since they hurt modularity.
- **Constant memory** — read-only by the device (R/W by host), scope is
  all grids, lifetime is the whole application. Physically resides in
  global memory but is cached, giving extremely fast, highly parallel
  access *if* access patterns are favorable. Hard-capped at 65,536
  bytes total per application.

CPU vs. GPU register architecture differs because of their different
scheduling models (sidebar, tying back to Chapter 4): CPUs save/restore
registers on every context switch, so register files stay small; GPUs
need zero-overhead switching between resident warps, so they hold
*all* resident threads' registers simultaneously, requiring a much
larger, dynamically-partitioned register file.

## 5.3 Tiling for reduced memory traffic (p. 104)

Core tradeoff: global memory is large but slow; shared memory is small
but fast. **Tiling** partitions the global-memory data into
shared-memory-sized subsets (*tiles*) that a block's threads
collaboratively load once and then reuse many times — valid only when
the kernel's computation on each tile can proceed independently of
other tiles.

Using the Chapter 3 naive matmul kernel as the running example: Fig.
5.6 shows that neighboring threads in a block redundantly re-fetch the
same `M` row and `N` column elements from global memory. If threads
collaborate to load each shared element only once, memory traffic
drops by a factor equal to the block (tile) width — e.g. 32× for
32×32 tiles. The tiled algorithm's core idea: divide the dot-product
computation into **phases**, one per tile; in each phase, every thread
in the block cooperatively loads one `M` element and one `N` element
into shared-memory arrays (conventionally named `Mds`/`Nds`), then all
threads use those shared values repeatedly for that phase's partial
dot product (Figs. 5.7–5.8). Because each phase's shared-memory
contents get reused by multiple threads before being replaced, this
exhibits **locality** — the same principle that makes CPU caches
effective, though GPUs implement it explicitly via shared memory/
registers rather than implicitly via hardware caching (a GPU SM runs
many concurrent threads competing for cache space, making an implicit
cache far less reliable than on a CPU core that runs one or two threads
at a time).

## 5.4 A tiled matrix multiplication kernel (p. 108)

Presents the full kernel (Fig. 5.9): `__shared__` arrays `Mds`/`Nds`
sized `TILE_WIDTH × TILE_WIDTH`; each thread loads one `M` element and
one `N` element per phase using its `Row`/`Col` indices offset by the
phase number `ph`, then a `__syncthreads()` ensures every thread's load
is visible before any thread starts computing with the tile. An inner
loop over `k < TILE_WIDTH` accumulates into a private `Pvalue`
register, after which a second `__syncthreads()` ensures every thread
is done *reading* the tile before the next phase's loads can overwrite
it. These two barriers illustrate two classic dependence types:
**read-after-write** (a *true* dependence — the barrier after loading)
and **write-after-read** (a *false* dependence — the barrier after
computing, preventing the next phase's loads from racing ahead). The
loop-over-phases structure is an instance of **strip-mining**: breaking
one long loop into an outer loop over phases and an inner loop that
does a bounded chunk of the original work each phase, so that barriers
can be placed around each chunk.

Benefit: tiling with `TILE_WIDTH`-sized tiles reduces global memory
traffic by a factor of `TILE_WIDTH`. For 32×32 tiles this raises the
kernel's ratio from 0.25 to 8 FLOP/B — still memory-bound relative to
the H100's 20.0 FLOP/B threshold, but enough to lift achievable
throughput from 0.84 TFLOPS to an estimated 26.8 TFLOPS (40% of the
66.9 TFLOPS peak). Pushing further into compute-bound territory needs
additional techniques (Chapter 15); in practice, production code would
reach for libraries like cuBLAS/CUTLASS rather than hand-rolling this
further. The kernel as presented assumes `Width` is a multiple of
`TILE_WIDTH` and that matrices are square — both relaxed in §5.5.

## 5.5 Boundary checks (p. 112)

Extends the kernel to arbitrary (non-tile-multiple) matrix widths.
Figs. 5.11–5.12 show that when `Width` isn't a multiple of
`TILE_WIDTH`, some threads' load indices land past the end of a row or
column in *every* phase, not just the last one — this can silently
alias onto the wrong data (an out-of-bounds row read wraps into the
next row of the linearized array) or access memory outside the
allocation entirely (undefined behavior). The fix follows one rule:
every access needs a matching bounds check.

- **Loading `M`:** only load if `Row < Width && (ph*TILE_WIDTH+tx) <
  Width`; otherwise store `0.0f` into shared memory — a value that
  contributes nothing to the dot-product sum, so it's always safe.
- **Loading `N`:** symmetric check, `(ph*TILE_WIDTH+ty) < Width && Col
  < Width`.
- **Storing `P`:** only write if `Row < Width && Col < Width` — a
  thread with no valid output element must still participate in
  loading tiles for its block's *other* threads to use, but must not
  write a result of its own.

With these three checks (Fig. 5.13) the kernel handles arbitrary square
matrix sizes; generalizing further to non-square, rectangular `j×k`
times `k×l` multiplication just requires replacing the single `Width`
parameter with separate `j`, `k`, `l` dimensions (left as an exercise).

## 5.6 Impact of memory usage on occupancy (p. 115)

Extends Chapter 4's occupancy discussion (which focused on registers)
to **shared memory** as a second resource that can cap occupancy.
Worked example: an H100 SM offers up to 228 KB of shared memory and
2048 thread slots, so fully using both requires averaging ≤114
B/thread of shared memory. The tiled matmul kernel uses only 8
B/thread (independent of `TILE_WIDTH`, since `Mds`+`Nds` together scale
with threads-per-block the same way), so it isn't shared-memory-limited.
Contrast: a hypothetical kernel using 38 KB of shared memory per
256-thread block averages 152 B/thread — exceeding the 114 B/thread
budget — so each SM maxes out at 1536 of its 2048 thread slots, capping
occupancy at 75%.

Some devices let a kernel reconfigure the SM to trade cache capacity
for more shared memory capacity, and shared-memory capacity itself
varies by device generation, so host code can query it at runtime via
`cudaGetDeviceProperties()`'s `devProp.sharedMemPerBlock` field rather
than assuming a fixed size. However, the straightforward
`__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];` declaration bakes
`TILE_WIDTH` in as a compile-time constant — to size shared memory
*at runtime* instead, declare it as `extern __shared__` with no size
in the declaration, merge `Mds`/`Nds` into one linearized array, and
pass the desired byte size as the kernel launch's third `<<<...>>>`
configuration argument (Fig. 5.14) — letting one compiled binary adapt
its shared-memory footprint to whatever the running device (or desired
occupancy tradeoff) calls for, without recompiling.

## 5.7 Summary (p. 118)

A kernel's execution speed is governed by its compute-to-global-
memory-access ratio: high ratios let a kernel approach the hardware's
peak compute throughput; low ratios leave it limited by memory
bandwidth. CUDA's registers, shared memory, and constant memory are
much smaller than global memory but much faster, and placing data in
them (rather than re-fetching from global memory) raises that ratio —
at the cost of redesigning the algorithm. **Tiling** is the chapter's
central strategy for this: using barrier synchronization to force a
block's threads to jointly focus on one data subset per phase so that
subset can live in fast on-chip memory. These special memories are
limited in size, and exceeding their capacity reduces the number of
threads that can simultaneously reside in an SM — hurting both compute
throughput and latency tolerance — so reasoning about these hardware
limits is a core skill of parallel programming. Though introduced here
via GPU/CUDA memory types, tiling and data locality are general
techniques that matter for high performance on any system with a
memory hierarchy, including multi-core CPUs.

## 5.8 Exercises (p. 119)

Twelve problems applying the chapter's concepts by hand: whether/how
shared memory helps vector addition (Q1); drawing the tiling diagram
for different tile and matrix sizes (Q2); what breaks if a
`__syncthreads()` is omitted (Q3); registers vs. shared memory tradeoffs
(Q4); computing bandwidth reduction for a given tile size (Q5); how
many instances of a local vs. shared variable exist across a kernel's
execution (Q6–7); global memory request counts with/without tiling
(Q8); compute- vs. memory-bound classification given hardware specs
(Q9); and multi-part occupancy/correctness analyses of given kernel
code (Q10–12). Not implemented in this repo's samples (end-of-chapter
exercises are out of scope — see the repo root README).
