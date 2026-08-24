# Chapter 5: Memory architecture and data locality

Source: *Programming Massively Parallel Processors*, 5th ed., Ch. 5 (pp. 93-121).

| File | Book section | Technique |
|------|-------------|-----------|
| `01_tiled_matrix_multiplication.cu` | §5.4 | `matrixMulTiledKernel` (Fig. 5.9): shared-memory tiled matmul for `Width` an exact multiple of `TILE_WIDTH`, timed against the §3.4 naive kernel for context |
| `02_tiled_matmul_boundary_checked.cu` | §5.5 | `matrixMulTiledBoundaryCheckedKernel` (Fig. 5.13): the same tiled kernel generalized with boundary checks for arbitrary `Width` |

§5.1 (the roofline model and compute-to-global-memory-access ratio) and §5.2
(CUDA's memory types -- registers, local, shared, constant, global -- and
their scope/lifetime/declaration syntax) are conceptual background with no
standalone kernel listing of their own, so they are summarized here rather
than given a sample file. §5.6 (impact of memory usage on occupancy and
dynamically-sized shared memory via `extern __shared__`) is likewise
conceptual and not exercised by a sample.

- A kernel is *compute-bound* if its performance is limited by peak FLOPS,
  *memory-bound* if limited by peak memory bandwidth; which regime a kernel
  falls into is set by its compute-to-global-memory-access ratio (FLOP/B)
  relative to the hardware's own FLOPS-to-bandwidth ratio (§5.1). The naive
  matmul kernel of Fig. 3.11 has a ratio of 2 FLOP / 8 B = 0.25 FLOP/B --- deep
  in memory-bound territory on hardware like the H100 (threshold ~20 FLOP/B)
  --- because every thread re-reads a full row of `M` and column of `N` from
  global memory with no reuse across threads (§5.1).
- CUDA exposes several on-chip/off-chip memory types with different
  scope, lifetime, and speed: automatic scalars live in per-thread
  *registers* (fastest, but limited and shared with occupancy); automatic
  arrays default to per-thread *local memory* (physically in global memory
  unless the compiler can register-allocate them); `__shared__` variables
  are per-block, on-chip, and shared by all threads in the block; `__constant__`
  variables are global-memory-backed but cached for fast read-only access
  from all threads of all grids; and plain `__device__` variables are
  ordinary global memory, visible everywhere, slow (§5.2).

## §5.3-§5.4 Tiled matrix multiplication -- `01_tiled_matrix_multiplication.cu`

§5.3 observes that in the naive kernel, threads within a block have heavily
overlapping input needs: every thread in the same block-row reads the same
row of `M`, and every thread in the same block-column reads the same column
of `N`. Tiling exploits this by having the block collaboratively stage a
`TILE_WIDTH x TILE_WIDTH` tile of `M` and one of `N` into `__shared__` memory
once, then letting every thread in the block reuse those on-chip tiles
`TILE_WIDTH` times each -- cutting the traffic to global memory for `M` and
`N` by a factor of `TILE_WIDTH`.

`matrixMulTiledKernel` (§5.4, Fig. 5.9) implements this as strip-mining: the
`Width`-long dot product is broken into `Width/TILE_WIDTH` phases. In each
phase `ph`:

1. Every thread loads one element of `M` (`M[Row][ph*TILE_WIDTH+tx]`) and one
   element of `N` (`N[ph*TILE_WIDTH+ty][Col]`) into the block's `Mds`/`Nds`
   shared-memory tiles.
2. `__syncthreads()` -- a **read-after-write** (true dependence) barrier: no
   thread may start consuming `Mds`/`Nds` until every thread has finished
   writing its share of them.
3. Each thread accumulates `TILE_WIDTH` products from the shared-memory
   tiles into its private `Pvalue` register.
4. `__syncthreads()` -- a **write-after-read** (false dependence) barrier: no
   thread may start overwriting `Mds`/`Nds` with the next phase's tile until
   every thread is done reading the current one.

After all phases, each thread writes its accumulated `Pvalue` to `P[Row][Col]`.
This file assumes `Width` is an exact multiple of `TILE_WIDTH` (32, the value
used in the book's own worked example in §5.6) -- the simplifying assumption
§5.4 states explicitly and §5.5 removes.

The sample also runs the §3.4 naive kernel (`matrixMulNaiveKernel`, no shared
memory) on the identical inputs and launch configuration for timing context.
Both kernels are warmed up with one discarded launch each before the timed
region (the Task 3 lesson noted in this project's brief: without a warm-up,
one-time PTX->SASS JIT cost can dominate whichever kernel is launched first
and produce a misleading comparison). **PASS/FAIL is decided purely by
whether both kernels agree with the CPU reference**; the naive-vs-tiled
timing is informational only, per the chapter brief (this file's job is
correctness + timing, not a head-to-head race the book itself makes). At
`Width = 1024`, on this repo's GPU the tiled kernel measures ~0.33 ms versus
the naive kernel's ~0.43 ms (~1.3x), consistent across repeated runs; the
gap is smaller than the book's factor-of-`TILE_WIDTH` traffic reduction would
suggest because a 1024x1024 float matrix (4 MB) comfortably fits in this
GPU's large L2 cache, so the naive kernel's repeated re-reads are served from
cache rather than DRAM for a large fraction of its accesses -- the sample
demonstrates the same *algorithmic* traffic reduction the book describes,
just measured on hardware whose cache narrows its real-world effect.

## §5.5 Boundary checks -- `02_tiled_matmul_boundary_checked.cu`

§5.5 extends the kernel to `Width` values that are *not* multiples of
`TILE_WIDTH`. Without a fix, threads in the last phase along either axis
either read the wrong (but in-bounds) element -- because linearized
row-major layout wraps a past-end-of-row index into the start of the next
row -- or read genuinely out-of-bounds memory. And this isn't confined to
literally the last phase or to threads that own a valid output element:
Fig. 5.12's `block1,1`, phase 0 example shows a thread with no valid `P`
element of its own still needing to load a tile element that its
neighbors *do* need.

The book's rule (§5.5): every global memory access gets its own bounds
check.

- Load `M[Row][ph*TILE_WIDTH+tx]` only if `Row < Width &&
  (ph*TILE_WIDTH+tx) < Width`; otherwise store `0.0f` into `Mds` (a value
  that contributes nothing to the accumulated dot product).
- Load `N[ph*TILE_WIDTH+ty][Col]` only if `(ph*TILE_WIDTH+ty) < Width &&
  Col < Width`; otherwise store `0.0f` into `Nds`.
- Write `P[Row][Col]` only if `Row < Width && Col < Width`.

`matrixMulTiledBoundaryCheckedKernel` (Fig. 5.13) is otherwise identical in
structure to §5.4's kernel -- same phase loop, same two `__syncthreads()`
barriers. This file keeps the kernel's original square-`Width` signature;
the book notes (end of §5.5) that generalizing further to rectangular
`j x k` by `k x l` matrices is "left as an exercise," so it isn't attempted
here.

Tested against three square sizes chosen specifically to *not* be multiples
of `TILE_WIDTH` (32), so the boundary paths are actually exercised rather
than accidentally dead code:

- `Width = 500` (`15*32 + 20`): several full tile phases, then one partial
  phase.
- `Width = 33` (`1*32 + 1`): one full phase, then a second phase that is
  almost entirely zero-padding.
- `Width = 15` (smaller than a single tile): the kernel's one and only phase
  is boundary-checked in both directions.

Each case computes a CPU reference with the same triple-loop inner-product
formula and checks the GPU result against it with `nearlyEqual`; PASS
requires all three sizes to match.

Build and run all samples in this chapter:

```sh
make run
```
