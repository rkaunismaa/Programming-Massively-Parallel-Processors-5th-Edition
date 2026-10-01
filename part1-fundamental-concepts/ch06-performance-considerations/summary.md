# Chapter 6 — Performance Considerations

Section-by-section summary of the book content. Page numbers are printed
book pages (Chapter 6 runs pp. 123–155).

## 6.1 Global memory access coalescing (p. 124)

Global memory is DRAM, and reading a DRAM cell is intrinsically slow:
it relies on a tiny capacitor's charge triggering a sense amplifier
over a long bit line (tens of nanoseconds — see the "Why are DRAMs so
slow?" sidebar), far slower than a sub-nanosecond clock cycle. DRAM
compensates with **bursting**: each access activates many sensors in
parallel and returns a whole range of consecutive locations at once
(a *DRAM burst*). Modern GPUs also sit SRAM-based L1/L2 caches in front
of DRAM, which move data in **cache-line**-sized chunks — so focused,
consecutive access is beneficial at both the DRAM-burst and
cache-line levels.

Because all threads in a warp execute the same load instruction at
once, hardware can detect when they target consecutive global memory
locations and **coalesce** — combine — them into one larger request.
Equally important is **alignment**: if the group's starting address
lines up with the start of a burst/cache line, the fewest possible
memory transactions are needed; a misaligned start can span extra
bursts and waste bandwidth.

Whether a warp's accesses coalesce depends on the linearized (row-major)
address math. Fig. 6.2 shows a matmul-style access to an `N` matrix
stored row-major, indexed `k*Width + col`, where `col` depends on
`threadIdx.x` — consecutive threads get consecutive `col`, hence
consecutive memory addresses: naturally coalesced. Fig. 6.3 shows the
same access pattern but with `N` stored **column-major**, indexed
`col*Width + k` — now consecutive threads' addresses are `Width`
elements apart, which is **not** coalesced (common when accessing a
matrix's transpose without physically transposing it).

**Corner turning** (Fig. 6.4) is the fix when a shared-memory tiling
kernel must load one matrix that's row-major and another that's
column-major: rather than mirroring the row-major loading scheme for
both (which makes the column-major loads non-coalesced), swap the
roles of `threadIdx.x`/`threadIdx.y` specifically for loading the
column-major matrix's tile, so that consecutive threads load
consecutive (same-column) elements from global memory — coalesced.
Once that tile is safely in shared memory (SRAM, not burst-sensitive),
each thread can access it in whatever order its computation needs with
little penalty, regardless of the layout it was loaded in. Coalescing
matters for **stores** too, and arguably more: an uncoalesced store can
require extra read-modify-write memory traffic, especially with ECC
enabled.

## 6.2 Hiding memory latency (p. 132)

Beyond bursting, DRAM systems add two more levels of parallelism:
**banks** and **channels**. A **channel** is a memory controller plus
bus connecting the processor to a set of DRAM **banks** (Fig. 6.7); a
processor typically has 1–8 channels. A bus's bandwidth is set by its
width and clock (DDR buses transfer on both clock edges) — e.g. a
64-bit DDR5-6400 bus delivers ~51.2 GB/s, far short of what a modern
GPU needs (1000+ GB/s), motivating High Bandwidth Memory (HBM), which
co-packages many wide, short buses with the processor.

Within one channel, a single bank's access-latency-to-transfer-time
ratio would badly underutilize the channel bus (Fig. 6.8(a)) — e.g. at
a 20:1 ratio, a 16 GB/s channel delivers only ~0.76 GB/s. The fix is
connecting multiple banks to each channel bus and **overlapping**
their access latencies (Fig. 6.8(b)): while one bank's cells are being
sensed, another bank can already be transferring its previously-ready
burst. To fully hide a latency-to-transfer ratio of R, a channel needs
more than R banks — partly for this overlap, and partly because more
banks reduce the odds of a **bank conflict** (two simultaneous accesses
targeting the same bank, which must then serialize).

Array elements are spread across channels and banks by hardware in an
**interleaved** pattern (Fig. 6.9) — one burst's worth of consecutive
elements per bank before moving to the next channel, wrapping through
all channels before returning to the next bank. This creates a direct
link between **parallel thread execution** and **parallel DRAM
organization**: maximizing occupancy (Chapter 4) doesn't just hide core
pipeline latency, it also generates enough simultaneous, coalesced
memory requests to spread work evenly across channels/banks and
actually realize the hardware's advertised memory bandwidth (worked
through with the tiled matmul example in Figs. 6.10–6.11, where
different thread blocks' tile loads land in different channels across
phases).

## 6.3 Vector loads and stores (p. 138)

Even a fully-occupied, fully-coalesced kernel issues one load/store
instruction per 4 bytes accessed per thread; **vector loads/stores**
let each thread access a larger contiguous chunk (e.g. 16 B) in a
single instruction, reducing instruction-execution overhead. Fig. 6.12
shows this applied to vector addition: casting the `x`/`y`/`z` pointers
to `float4` before dereferencing causes the compiler to emit one
128-bit vector load/store per four elements instead of four scalar
ones — a 75% cut in load/store instruction count for the same data
volume (two vector loads replace eight scalar loads). This is also
valuable when a kernel can't reach full occupancy: with fewer resident
warps available to overlap latency, issuing more memory traffic per
instruction becomes a direct way to still saturate memory bandwidth
(an example appears in Chapter 15). Handling array lengths not
divisible by the vector width needs ordinary scalar-load/store boundary
handling, left as an exercise.

## 6.4 Shared memory bank conflicts (p. 139)

Shared memory is SRAM, not DRAM, so it doesn't need bursting — instead
it's organized into 32 banks (matching warp size), each 4 bytes wide,
with consecutive 4-byte words placed in consecutive banks in round-robin
fashion. If all 32 threads of a warp access 32 different banks, all
accesses are served in one parallel pass; if multiple threads in a
warp land on the *same* bank, that's a **bank conflict**, and the
hardware serializes those requests into multiple passes.

Worked example: the §6.1 corner-turning code writes
`a[threadIdx.x][threadIdx.y]` into a `TILE_DIM × TILE_DIM` (= 32×32)
`__shared__` array. Because the warp shares one `threadIdx.y` and
varies `threadIdx.x`, and the array's row-major linear index is
`i*32+j`, consecutive threads' indices are 32 apart — all landing in
*the same* bank (32 mod 32 = 0 every time): a full 32-way bank
conflict. The fix is **padding**: declare the array with one extra
column (`[TILE_DIM][TILE_DIM+1]`), which shifts the linear index
formula to `i*33+j`, so consecutive threads now land in different
banks (0, 33 mod 32=1, 66 mod 32=2, …) — conflict-free. Padding isn't
free: it burns a little extra shared memory, and losing a power-of-two
row width means the linearized-index multiply can no longer be
optimized to a bit-shift. The benefit of padding also scales with the
original stride — a smaller original row width (e.g. 16, giving a
16-way rather than 32-way conflict pattern) sees comparatively less
benefit from the same fix.

## 6.5 Thread coarsening (p. 141)

Up to now, every kernel has assigned the smallest possible unit of work
to each thread (one output element). That maximizes transparent
scalability — if hardware has enough resources, all that work really
does run in parallel — but it also means fully paying whatever
*per-thread* overhead the parallelization itself introduces: redundant
data loading across thread blocks, synchronization overhead,
instruction overhead, etc. **Thread coarsening** means giving each
thread multiple units of work (the **coarsening factor**) via a
**coarsening loop**, so that if the hardware was going to serialize the
work anyway (not enough SMs/resources to run it all in parallel), the
overhead of parallelizing it is paid fewer times. A toy example:
multiplying the global index by a coarsening factor of 4 and looping 4
times inside the kernel instead of launching 4× as many threads. The
vector-load optimization in §6.3 is itself a form of coarsening (4
additions coarsened per thread to amortize instruction overhead); the
tiled matmul kernel (Chapter 5) is coarsenable too, since separate
thread blocks currently redundantly reload the same input tiles
(explored in depth in Chapter 15).

Three pitfalls: (1) coarsening computations that have no real
parallelization overhead to begin with buys nothing; (2) coarsening too
aggressively reduces the number of blocks exposed to the hardware below
what's needed to fill it (underutilization, possibly leaving a partial
tail wave — tying back to Chapter 4's wave/tail-effect discussion), and
the "right" coarsening factor becomes device- and dataset-specific,
undermining transparent scalability; (3) coarsening increases
per-thread resource usage (more registers and/or shared memory per
thread, since each thread does more), which can itself reduce occupancy
enough to outweigh the benefit.

## 6.6 Loop unrolling (p. 144)

**Loop unrolling** replicates a loop body N times while dividing the
iteration count by N, which helps tolerate latency in two ways: fewer
loop-back branch instructions (GPUs lack the CPU's sophisticated branch
prediction, so each branch instruction stalls until resolved), and more
independent instructions exposed for the compiler to reorder around
each other (**instruction scheduling**) — e.g. interleaving
`A(i+1)`/`A(i+2)`/`A(i+3)` between `A(i)` and `B(i)` so a thread has
useful work queued while waiting on `A(i)`'s result. Compilers apply
unrolling and scheduling automatically and aggressively, especially for
small, constant-bound loops; `#pragma unroll N` gives explicit control
when needed. Loop unrolling is essential for making **thread
coarsening** efficient: a coarsening loop's local accumulator array
(e.g. `int x[4]`) is normally placed in slow local/global memory
because it's accessed with a variable index — but once the small,
constant-bound coarsening loop is fully unrolled, every access becomes
constant-indexed, letting the compiler promote the array straight into
registers.

## 6.7 Double buffering (p. 146)

A common block-synchronized pattern has every thread read a value
another thread wrote, then write its own value for others to read next
iteration — needing two `__syncthreads()` calls per iteration: one
enforcing the unavoidable **read-after-write** (*true*) dependence, one
enforcing a **write-after-read** (*false*) dependence that only exists
because reads and writes share the same memory location across
iterations. **Double buffering** eliminates the false dependence (and
its barrier) by allocating two buffers and alternating (swapping)
which one is read-from vs. written-to each iteration, so a write can
never race ahead of a read it would have corrupted — removing one of
the two barriers per iteration entirely. More concrete applications
appear in later chapters (e.g. Chapter 11's prefix sum, Chapter 15's
matmul).

## 6.8 A checklist of optimizations (p. 147)

Consolidates every optimization covered in Part 1 of the book into one
checklist (Fig. 6.13), grouped into three categories by what aspect of
performance they target, plus one general-purpose entry:

- **Compute utilization** — keeping cores busy: occupancy tuning
  (Ch. 4/5/6.2), loop unrolling (§6.6), reducing control divergence
  (Ch. 4; strategies include rearranging thread/data assignment —
  Ch. 10, 18 — and rearranging data layout — Ch. 17).
- **Memory utilization** — reducing/speeding up memory traffic: using
  coalesceable global accesses (§6.1; strategies include corner
  turning, and staging irregular accesses through shared memory first,
  i.e. "packing" — Ch. 12, 13, 14), shared memory tiling (Ch. 5, 7, 8,
  16), register tiling (Ch. 3, 5, 8, 11, 15), vector loads/stores
  (§6.3), avoiding shared memory bank conflicts (§6.4; Ch. 11, 15), and
  privatization (not yet introduced — previewed here, covered in
  Ch. 9, 12, 18).
- **Synchronization latency** — reducing time spent waiting at
  barriers: warp-level primitives (Ch. 10, 11, 12, 15) and double
  buffering (§6.7; Ch. 11, 15).
- **General**: thread coarsening (§6.5) — its category depends on
  context (e.g. reduces redundant loading in Ch. 8/15, reduces
  privatization-copy overhead in Ch. 9, reduces sync/divergence
  overhead in Ch. 10/11, improves coalescing in Ch. 12/14).

The checklist is explicitly not exhaustive — other optimizations appear
where relevant to a specific pattern (e.g. constant memory for
convolution in Chapter 7).

## 6.9 Optimization strategy (p. 153)

Because optimizations trade one resource's pressure for another's, the
first step is always identifying the kernel's actual **bottleneck**
resource — applying an optimization that doesn't target the bottleneck
does nothing, and one that *increases* pressure on the bottleneck can
actively hurt performance (e.g. shared memory tiling helps when global
bandwidth is the bottleneck, but can make things worse if occupancy is
already constrained by shared memory usage). Optimization is therefore
**iterative**: identify the bottleneck (via profiling tools), apply a
targeted fix, re-identify the new bottleneck, repeat. Because
bottlenecks are often hardware-specific, the same kernel can need
different fixes on different devices. The **Roofline Model**
(Chapter 5) gives a stopping condition: compare a kernel's actual
measured throughput to the hardware's limit at its computational
intensity — a kernel already close to that limit has little room left
to improve.

## 6.10 Summary (p. 154)

This chapter covered the GPU's off-chip DRAM architecture and the
performance techniques that follow from it — coalescing and
latency-hiding via channel/bank parallelism — plus thread coarsening,
loop unrolling, and double buffering. Combined with Chapters 4–5, this
equips the reader to reason about the performance of essentially any
kernel. The chapter (and Part 1 of the book) closes with the checklist
of common optimizations (§6.8), applied repeatedly to real patterns and
case studies in Parts 2 and 3.

## 6.11 Exercises (p. 154)

Five problems applying the chapter's concepts by hand: classifying a
multi-array kernel's accesses as coalesced/uncoalesced/not-applicable
(Q1); writing a corner-turning matmul kernel (Q2); for which
`BLOCK_SIZE` values the Chapter 5 tiled kernel fully avoids
uncoalesced accesses (Q3); implementing boundary-correct vector
loads/stores (Q4); and computing shared-memory bank conflicts for
various strided access patterns (Q5). Not implemented in this repo's
samples (end-of-chapter exercises are out of scope — see the repo root
README).
