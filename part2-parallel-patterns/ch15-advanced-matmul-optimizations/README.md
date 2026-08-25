# Chapter 15: Advanced optimizations for matrix multiplication

Source: *Programming Massively Parallel Processors*, 5th ed., Ch. 15 (pp. 349-370).

| File | Book section | Technique |
|------|-------------|-----------|
| `01_matmul_coarsened_larger_tiles.cu` | §15.3 | Thread-coarsened tiled matmul with a 128x128 block-level output tile and an 8x8 per-thread output tile (Figs. 15.3-15.7) |
| `02_matmul_register_tiled.cu` | §15.4 | Register tiling of the input tiles: loop-interchanged `mm()` loads each shared-memory strip once per thread instead of once per output element (Fig. 15.9) |
| `03_matmul_coalesced_output_store.cu` | §15.5 | Warp/quadrant/lane output-tile rearrangement + `float4` vector stores for coalesced writes (Fig. 15.10) |
| `04_matmul_bank_conflict_free.cu` | §15.6 | `A_s` padded to a `bK+1` leading dimension to eliminate the 8-way shared-memory bank conflict on its strided strip load (Fig. 15.11) |
| `05_matmul_software_pipelined.cu` | §15.8/§15.9 | Double-buffered software pipelining via `cuda::memcpy_async`/`cp.async` (Fig. 15.14) — **requires compute capability >= 8.0** |

§15.1 (background on GEMM and vector-matrix multiplication), §15.2 (data
reuse analysis motivating larger tiles), §15.7 (occupancy considerations —
why register pressure limits these kernels to ~256 threads/SM and how
vector memory ops + aggressive unrolling compensate), and §15.9's remaining
material beyond `cuda::memcpy_async` (cuBLAS/cuDNN/CUTLASS, tensor cores,
WMMA/WGMMA, TMA) are conceptual/background discussion with no standalone
kernel of their own to implement, so they are summarized here rather than
given a sample file.

- §15.1: matrix multiplication as the compute-heavy core of GEMM
  (`D = αA×B + βC`); this chapter's optimizations target the `A×B` step.
- §15.2: for an `m×n` output tile with inner dimension `k`, arithmetic
  intensity is `0.5·m·n/(m+n)` operations/byte — independent of `k`, and
  growing with larger `m`, `n`. Larger output tiles let the same input
  panels of `A`/`B` be reused across more output elements before eviction,
  raising arithmetic intensity (§15.2's H100 example: 8 FLOP/B at 32x32
  tiles vs. 32 FLOP/B at 128x128 tiles).
- §15.7: an 8x8 register-tiled output tile needs 64 registers just for
  `C_r`, plus 8+8 more for register-tiled input strips — at the hardware
  255-register/thread cap, only 256 threads/SM fit (12.5% of a modern SM's
  2048-thread max occupancy). The two drawbacks of this low occupancy
  (fewer memory instructions in flight, less latency-hiding) are addressed
  by vector loads/stores (§15.3/§15.5) and aggressive loop unrolling
  (applied throughout `01`-`05` via `#pragma unroll`) respectively; software
  pipelining (§15.8, file `05`) addresses the remaining barrier-induced
  memory/compute phase separation.
- §15.9 beyond `cuda::memcpy_async`: production GEMM is normally done via
  cuBLAS/cuDNN/CUTLASS, which use dedicated tensor-core hardware (WMMA on
  pre-Hopper, WGMMA on Hopper+, reading operands straight from shared
  memory) and, on Hopper+, the Tensor Memory Accelerator (TMA) for
  asynchronous whole-tensor transfers — a further generalization of the
  LDGSTS/`cuda::memcpy_async` mechanism file `05` uses for individual tile
  elements.

## §15.3 Thread coarsening with larger tiles — `01_matmul_coarsened_larger_tiles.cu`

Reimplements Chapter 5's tiled matmul (no cross-chapter include, per this
repo's convention) at the book's larger-tile granularity: each block
computes a `bM×bN = 128×128` block-level output tile with `bK = 8`, using
256 threads; since 256 threads can't cover 128×128 = 16384 output elements
1:1, each thread is coarsened to an 8×8 (`tM×tN`) thread-level output tile,
kept in registers (`C_r`) via `#pragma unroll`-forced constant indexing.
`clear()`, `loadTile()`, `mm()`, and `writeTile()` are direct
transcriptions of Figs. 15.4-15.7 (`mm()` here uses the *basic*, non-
register-tiled loop order); the main kernel is Fig. 15.3 verbatim, modulo
macro names. Tested against a `200×180×90` case (deliberately not a
multiple of `bM`/`bN`/`bK`, exercising every boundary-check path in
`loadTile`/`writeTile`) and a `1024×1024×1024` performance case.

## §15.4 Register tiling of the input tiles — `02_matmul_register_tiled.cu`

Identical to `01` except for `mm()`. In `01`, each element of a thread's
`tM×bK`/`bK×tN` shared-memory input sub-tile is re-read from shared memory
once per output element that consumes it (row/col outer, `k` inner). Fig.
15.9 interchanges the loop nest so the `k` (inner) dimension becomes
outermost: for each of the `bK` strips, the thread loads one row-strip of
`A` (`a_r[tM]`) and one column-strip of `B` (`b_r[tN]`) from shared memory
into registers exactly once, then reuses those register values to update
all `tM×tN` output elements before moving to the next strip — removing the
redundant shared-memory traffic `01` pays for. Same two test cases as `01`.

## §15.5 Coalesced storing of the output tile — `03_matmul_coalesced_output_store.cu`

Builds on `02`. The problem (§15.5): a contiguous 8×8 thread-level output
tile makes threads in the same warp write to `C` locations 8 elements
(32 B) apart — not coalesced. Fig. 15.10's fix, worked out here for the
book's own 128×128-block/256-thread example: the 8 warps are arranged 2×4
(each warp owning a 64×32 warp-level tile), each warp-level tile is split
into four 32×16 quadrants, and the warp's 32 threads are arranged 8×4 so
each thread takes a 4×4 sub-tile from *each* quadrant — four physical 4×4
sub-tiles per thread instead of one contiguous 8×8 tile. A whole sub-tile
row (4 floats = 16 B) now fits one `float4` vector store, and the four
threads sharing a quadrant row (varying lane-column) issue vector stores to
adjacent 16 B chunks — coalesced. `mm()` (still register-tiled, Fig. 15.9)
is invoked once per quadrant against that quadrant's own row/column offset
into `A_s`/`B_s`, exactly as §15.5 directs ("the `mm()` and `writeTile()`
device functions need to be revised to iterate through these 4×4 sub-tiles
... in each step"). The vector-store path falls back to a scalar,
per-element boundary-checked store whenever a quadrant sub-tile isn't fully
in-bounds or its destination address isn't 16 B aligned, so the kernel
stays correct for arbitrary `M`/`N`. Tested against `01`/`02`'s two cases
plus a third, `131×173×67`, whose `N` is not a multiple of 4 — deliberately
exercising the scalar fallback for the trailing unaligned columns.

## §15.6 Eliminating bank conflicts — `04_matmul_bank_conflict_free.cu`

Builds on `03`; the only change is `A_s`'s shared-memory layout. With the
8×4 lane arrangement from §15.5, the four threads at a fixed lane-row all
read the same row of `A_s` in a given strip, and consecutive lane-row
groups (0, 4, 8, ..., 28 rows apart) land `4*bK`, `8*bK`, ... linear
elements apart. At `bK = 8` this is a multiple of 32 for every group — an
8-way bank conflict on every strip load, exactly the arithmetic the book
works through in Fig. 15.11(a). Fig. 15.11(b)'s fix: pad `A_s` to a leading
dimension of `bK+1 = 9` instead of `bK`; the same row offsets now land
`4*9=36`, `8*9=72`, ... elements apart, which mod 32 spread across 8
distinct banks — conflict-free. The change is exactly as small as the book
states: `A_s` becomes `bM*(bK+1)` floats, and `bK+1` is passed as its
leading dimension everywhere it's loaded from or read; `B_s` is left
unpadded, per the book's note that its strip load (32 consecutive elements
per warp) already spans all 32 banks. Same three test cases as `03`.

## §15.8/§15.9 Software pipelining — `05_matmul_software_pipelined.cu` (compute capability >= 8.0 required)

Builds on `04`. §15.7 observes that with only one thread block resident per
SM (a consequence of the register pressure discussed there), a block
executes in two separated phases — memory-bound (loading tiles, ALUs idle)
and compute-bound (`mm()`, memory hardware idle) — because a barrier
enforces a *false* dependence between finishing computation with the old
tile and overwriting it with the new one in the same buffer. Fig. 15.14
removes that false dependence with double buffering (`Acurr_s`/`Bcurr_s`
for compute, `Anext_s`/`Bnext_s` for the concurrently-issued next load),
letting the compiler interleave the two phases' instructions once inlined.

This file combines that double-buffering structure with §15.9's
`cuda::memcpy_async` (lowering to the hardware `cp.async`/LDGSTS
instruction on Ampere+) instead of the book's plain, implicitly-synchronous
`loadTile()`: each thread issues its tile-load elements as asynchronous
copies from global memory directly into shared memory, tracked by a
thread-scoped `cuda::pipeline`, so the copy genuinely executes in the
background while this thread's own `mm()` FMA instructions for the current
tile are in flight — true producer/consumer overlap, not just a dependence
the compiler might exploit. This mirrors this repo's
`ch06/06_double_buffering_async_copy.cu`, which applies the same
double-buffering + `cuda::pipeline` combination to Chapter 5's plain tiled
matmul; here it's layered on top of every optimization from this chapter
(register tiling, coalesced quadrant output, bank-conflict-free padding).
Each thread issues tile `t+1`'s load into the *other* buffer before waiting
on / computing with tile `t`'s own data, so the async-copy engine has the
whole iteration's compute time to deliver the next tile; `pipe.consumer_
wait()` (this thread's own copies) is followed by `__syncthreads()` (all
threads' copies) before the shared tile is read, matching Fig. 15.14's
single per-iteration barrier.

Because `cuda::memcpy_async` copies a fixed, unconditional byte span (no
per-element bounds check like `loadTile()`), this file — like
`ch06/06` — requires `M`, `N`, `K` to be exact multiples of `BM`, `BN`,
`BK` and does not attempt the general boundary-checked case; it is tested
at `M=N=K=1024`.

**This is the only file in this chapter that requires compute capability
>= 8.0** (`cuda::memcpy_async` targeting shared memory needs Ampere's
hardware `cp.async` support) and is compiled with `-arch=sm_80` by this
chapter's `Makefile`, unlike files `01`-`04` which build with `-arch=sm_75`.
At startup it queries `cudaGetDeviceProperties` on whatever device it
actually runs on; if that device's compute capability is below 8.0, it
prints a clear message and exits with status 0 instead of crashing.

Build and run all samples in this chapter:

```sh
make run
```

This machine has two GPUs: an RTX 2070 SUPER (compute capability 7.5) and
an RTX 4090 (compute capability 8.9). Note that CUDA's own device
enumeration on this machine (fastest-first, the default when
`CUDA_DEVICE_ORDER` is unset) numbers the RTX 4090 as device 0 and the RTX
2070 SUPER as device 1 — the *opposite* of `nvidia-smi`'s PCI-bus-order
numbering. `make run` (no `CUDA_VISIBLE_DEVICES` override) therefore
already lands file `05` on the RTX 4090 by default; this matches
`ch06/06_double_buffering_async_copy.cu`'s existing behavior in this repo.
