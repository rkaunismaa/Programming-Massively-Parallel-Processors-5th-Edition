# Chapter 6: Performance considerations

Source: *Programming Massively Parallel Processors*, 5th ed., Ch. 6 (pp. 123-155).

| File | Book section | Technique |
|------|-------------|-----------|
| `01_coalesced_vs_uncoalesced_access.cu` | §6.1 | Coalesced (Fig. 6.2) vs. un-coalesced (Fig. 6.3) global memory access pattern |
| `02_vectorized_loads_float4.cu` | §6.3 | `float4` vector loads/stores (Fig. 6.12) vs. scalar vector addition |
| `03_shared_memory_bank_conflicts.cu` | §6.4 | Conflicting `a[TILE_DIM][TILE_DIM]` shared-memory store vs. padded `a[TILE_DIM][TILE_DIM+1]` |
| `04_thread_coarsening.cu` | §6.5 | Thread coarsening applied to the Ch. 5 tiled matmul: one M tile reused across `COARSE_FACTOR` N tiles |
| `05_loop_unrolling.cu` | §6.6 | `#pragma unroll` vs. `#pragma unroll 1` on a tiled matmul's inner accumulation loop |
| `06_double_buffering_async_copy.cu` | §6.7 | Double-buffered shared-memory tile loading via SM80+ `cuda::memcpy_async`/`cuda::pipeline` (cp.async) — **requires compute capability >= 8.0** |

§6.2 (hiding memory latency via DRAM banks/channels and occupancy) and §6.8-6.9
(the optimization checklist and strategy, which point forward to techniques
demonstrated throughout the rest of the book) are conceptual/background
material with no standalone kernel of their own, so they are summarized here
rather than given a sample file.

- §6.2: DRAM achieves its advertised bandwidth only when many banks and
  channels are kept simultaneously busy (interleaved data distribution,
  Fig. 6.9); maximizing occupancy (§4-era topic) matters again here because
  it's what generates enough concurrent memory requests to spread across
  those banks/channels and hide each individual access's long latency.
- §6.8-6.9: a consolidated checklist of the optimizations covered across
  Ch. 4-6 (occupancy tuning, loop unrolling, reducing control divergence,
  memory coalescing, shared-memory tiling, register tiling, minimizing
  divergent branching, tuning parallelism granularity via thread
  coarsening) and a strategy for applying them, both of which point forward
  to Parts 2 and 3 of the book rather than introducing new mechanics.

## §6.1 Coalescing — `01_coalesced_vs_uncoalesced_access.cu`

The book's example (Fig. 6.2 vs Fig. 6.3): a matrix `N` used as matmul's
second input. When `N` is row-major, the index `k*Width+col` makes
consecutive threads (consecutive `col`) read consecutive memory locations —
coalesced. When `N` is column-major instead (e.g. because it's really the
transpose of a row-major matrix being accessed in place), the index
`col*Width+k` makes consecutive threads read locations `Width` elements
apart — un-coalesced.

This file isolates that exact contrast: both kernels compute the column
sums of the same logical `Width x Width` matrix, reading from two physical
buffers holding the same logical values in different layouts (`N_row`,
row-major; `N_col`, column-major — literally the transpose of `N_row`
flattened). The coalesced kernel indexes `N_row` with `k*Width+col`
(Fig. 6.2's exact expression); the uncoalesced kernel indexes `N_col` with
`col*Width+k` (Fig. 6.3's exact expression). Same logical result, same
amount of work — only the access pattern differs. At `Width = 4096` on this
repo's GPU: coalesced ~0.12 ms vs. uncoalesced ~0.42 ms (~3.6x), stable
across repeated runs.

## §6.3 Vector loads/stores — `02_vectorized_loads_float4.cu`

Fig. 6.12's kernel casts the `x`/`y`/`z` array pointers to `float4` before
dereferencing, so each thread issues a single 16 B vector load/store
instead of four separate 4 B scalar ones — two vector loads replace eight
scalar loads for the same data volume (a 75% cut in load-instruction count,
per the text directly under Fig. 6.12). `n` is chosen as an exact multiple
of 4 here, sidestepping the boundary condition the book explicitly leaves
as an exercise. At `n = 2^24` elements on this repo's GPU: scalar ~0.22 ms
vs. float4 ~0.20 ms (~1.1x) — a modest gain, expected on a bandwidth-rich
GPU like the 4090 where this kernel is already memory-bandwidth-bound at
full occupancy; the instruction-count reduction the book describes matters
most when a kernel *can't* reach full occupancy or issue enough concurrent
requests otherwise (the book itself notes vector loads matter more in such
cases, pointing to Chapter 15).

## §6.4 Shared memory bank conflicts — `03_shared_memory_bank_conflicts.cu`

The book's worked example: `__shared__ float a[TILE_DIM][TILE_DIM]` (32x32)
with the store `a[threadIdx.x][threadIdx.y] = ...` puts threads 0, 1, 2, ...
of a warp at linear indices 0, 32, 64, ... — all bank 0 (`32 mod 32 == 0`),
a 32-way bank conflict the hardware must serialize. Padding to
`a[TILE_DIM][TILE_DIM+1]` shifts those same threads to linear indices
0, 33, 66, ... — banks 0, 1, 2, ... (`33 mod 32` cycling), conflict-free.

This file embeds that exact `a[threadIdx.x][threadIdx.y]` pattern inside a
full shared-memory matrix-transpose kernel (the classic vehicle for it):
each block coalesced-loads a tile from global memory, stores it into shared
memory with the strided (conflicting, or padded/conflict-free) pattern,
then coalesced-stores it back out transposed. Both variants are checked
against a CPU transpose. At `Width = 4096` on this repo's GPU: conflicting
~0.167 ms vs. padded ~0.164 ms (~1.02x) — real but small, stable across
repeated runs. The gap is modest because this kernel's overall time is
dominated by its 128 MB of global memory traffic (64 MB read + 64 MB
write) rather than by the ~200-cycle serialization added by the shared
memory conflict itself; the *mechanism* the book describes (32-way
serialized shared-memory access vs. one access per cycle) is real and
reproduced here, it's just not the kernel's bottleneck at this problem
size on this hardware.

## §6.5 Thread coarsening — `04_thread_coarsening.cu`

The book names this exact example: "the tiled matrix multiplication kernel
in Chapter 5 ... using fewer thread blocks and having each block process
larger output tiles enables each input tile to be loaded fewer times."
This file reimplements Ch. 5's tiled matmul locally (no cross-chapter
include) and adds a `COARSE_FACTOR = 4` coarsened variant: each block still
loads one `M` tile per phase, but now reuses it across `COARSE_FACTOR`
separate `N` tiles / output tiles laid out along the column direction
(accumulated into `COARSE_FACTOR` per-thread registers), instead of
`COARSE_FACTOR` separate thread blocks each redundantly loading their own
copy of the same `M` tile. The coarsened kernel therefore launches with
`COARSE_FACTOR` fewer blocks along the grid's x dimension — that reduced
block count *is* the optimization the book describes, not an unfair setup;
block dimension and total output size are identical between the two.
`Width = 1024` and `COARSE_FACTOR = 4` are chosen so both grids
(1024 and 256 blocks respectively) comfortably exceed one wave on this
GPU's 128 SMs, avoiding the "coarsening too far causes a partial wave"
pitfall §6.5 itself warns against. At this size on this repo's GPU:
uncoarsened ~0.33 ms vs. coarsened ~0.32 ms (~1.04x) — a modest, stable
gain; redundant `M`-tile traffic here is largely absorbed by this GPU's
large L2 cache rather than hitting DRAM, which narrows the real-world
benefit of the traffic reduction the book describes (the same caveat
`ch05`'s README notes for the tiled-vs-naive comparison).

## §6.6 Loop unrolling — `05_loop_unrolling.cu`

Two Ch. 5-style tiled matmul kernels (reimplemented locally), identical
except for the pragma on the inner `TILE_WIDTH`-deep accumulation loop:
`#pragma unroll` (full unroll — the loop bound is a small compile-time
constant, the book's own description of a loop the compiler would unroll
fully) vs. `#pragma unroll 1` (unrolling explicitly disabled, exactly the
semantics the book states for setting the unrolling factor to 1). At
`Width = 1024` on this repo's GPU: unrolled ~0.33 ms vs. not-unrolled
~0.49 ms (~1.48x) — the largest, most reproducible gap of any comparison in
this chapter, consistent with the book's claim that loop unrolling reduces
branch-instruction stalls and exposes more independent instructions for
scheduling.

## §6.7 Double buffering — `06_double_buffering_async_copy.cu` (compute capability >= 8.0 required)

The book's toy example removes a false write-after-read dependence by
splitting a single shared buffer into `inBuffer`/`outBuffer` that swap each
iteration, eliminating the second `__syncthreads()` a single-buffer version
needs between reading and writing. This file applies the same idea to the
Ch. 5 tiled matmul's tile-loading step (reimplemented locally): two
physical `Mds`/`Nds` buffers ping-pong across phases, and goes one step
further than the book's toy example by using SM80+ hardware asynchronous
copy (`cuda::memcpy_async`, which lowers to the `cp.async` PTX instruction
on Ampere and newer, via a `cuda::pipeline<cuda::thread_scope_thread>`)
so the *next* phase's tile is fetched by the copy engine directly into
shared memory while the *current* phase's inner product is still being
computed on the other buffer — genuine overlap between the async-copy
engine and the compute pipeline, not just dependence elimination. Each
thread issues phase `ph+1`'s load via `producer_acquire`/`memcpy_async`/
`producer_commit` before waiting on (`consumer_wait`) and computing on
phase `ph`'s already-in-flight tile, then `consumer_release`s it; a
`__syncthreads()` after `consumer_wait()` is still required because each
thread's `consumer_wait()` only guarantees its own tile element landed, not
its block-mates'.

**This is the only file in this chapter that requires compute capability
>= 8.0** (`cuda::memcpy_async` targeting shared memory needs Ampere's
hardware `cp.async` support) and is compiled with `-arch=sm_80` by this
chapter's `Makefile`, unlike files 01-05 which build with `-arch=sm_75`.
At startup it queries `cudaGetDeviceProperties` on whatever device it
actually runs on; if that device's compute capability is below 8.0, it
prints a clear message and exits with status 0 instead of crashing with a
cryptic "no kernel image available for execution" error, so the sample is
self-diagnosing on hardware without an Ampere+ GPU. On this repo's RTX 4090
(compute capability 8.9) it runs and verifies correct at `Width = 1024`,
~0.36 ms — slightly slower than the plain tiled kernel in `04`/`05`
(~0.33 ms): at this problem size the per-thread, per-float `memcpy_async`/
pipeline bookkeeping overhead outweighs the benefit of overlapping a
128-thread-block's worth of 4 B transfers with compute, which is an honest
result rather than a case for hiding the number — the mechanism (real
`cp.async`-based prefetch with block-mate-safe synchronization) is what
this file demonstrates, not a guaranteed win at every problem size.

Build and run all samples in this chapter:

```sh
make run
```

Note: file 06 needs to actually execute on a compute capability >= 8.0
device to be meaningfully tested; on a machine where the only visible GPU
is below SM80 (e.g. this repo's other GPU, an RTX 2070 SUPER, compute
capability 7.5), it still builds cleanly (building `-arch=sm_80` code
needs no GPU at all) but will print the graceful skip message at run time
instead of exercising the async-copy kernel.
