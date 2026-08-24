# Chapter 8: Stencil computation

Source: *Programming Massively Parallel Processors*, 5th ed., Ch. 8 (pp. 183-198).

| File | Book section | Technique |
|------|-------------|-----------|
| `01_stencil_naive.cu` | §8.2 | Basic stencil sweep kernel (Fig. 8.6): one thread per output grid point, all 7 stencil taps read from global memory, boundary layer left untouched |
| `02_stencil_shared_memory_tiling.cu` | §8.4 | Shared-memory-tiled sweep kernel (Fig. 8.8): block sized to the (small, `t=8`) cubic *input* tile, one thread per input element loaded, only interior threads compute an output point |
| `03_stencil_thread_coarsening.cu` | §8.5 | Thread-coarsened sweep kernel (Fig. 8.10): 2D (`32x32`) thread block per x-y column, each thread loops over a whole range of output z-planes, keeping only 3 shared-memory planes (prev/curr/next) live at a time |
| `04_stencil_register_tiling.cu` | §8.6 | Register-tiled sweep kernel (Fig. 8.12): builds on file 03, but the previous/next z-planes move into per-thread registers, leaving only the current plane in shared memory |

All four files implement the same **3D 7-point (order-1) stencil** — the
book confirms this explicitly ("the kernel in Fig. 8.6 assumes a 3D grid and
a 3D 7-point stencil like the one in Fig. 8.3(c)", §8.2): the center grid
point plus one neighbor on each side along x, y, and z, weighted by 7
coefficients `c0..c6` (13 FLOP per output point: 7 multiplies + 6 adds,
confirmed by §8.3's arithmetic-intensity analysis). Coefficients are
hard-coded `#define`s, one of the two options §8.2 calls out ("hard-coded in
the code, or ... placed in constant memory").

Per §8.2's simplifying boundary assumption (Fig. 8.5), the outermost layer
of the `N x N x N` grid stores fixed boundary conditions and is **never
written** by any sweep — only the `(N-2)^3` interior points are computed.
Every test harness in this chapter pre-fills both the CPU reference buffer
and the GPU output buffer with a sentinel value before running the sweep,
then compares the *entire* `N^3` buffer: this checks the interior values AND
verifies the boundary layer was genuinely left untouched, in one pass.

§8.3 (memory bandwidth / arithmetic intensity) is a quantitative analysis
with no kernel listing of its own; its key numbers are folded into each
file's comments below rather than given a separate file.

## §8.1-8.2 Basic kernel — `01_stencil_naive.cu`

Each thread is assigned one 3D output grid point via the familiar
`blockIdx`/`blockDim`/`threadIdx` linear mapping (Fig. 8.6, lines 02-04), and
threads whose point falls in the boundary layer (any axis at index `0` or
`N-1`) are turned off. No shared memory, no coarsening — every one of the 7
stencil taps is a separate global-memory load. §8.3's arithmetic-intensity
analysis of this exact kernel: 13 FLOP for 7 loads + 1 store (32 B), i.e.
0.41 FLOP/B — far below the 1.625 FLOP/B ideal for a 3D 7-point stencil,
making it strongly memory-bound.

Block size is `8x8x8` (512 threads), the size the book's own end-of-chapter
exercises use for this kernel. Tested with grid sizes that are and aren't
multiples of the block dimension, exercising partial blocks at the grid's
far edges.

## §8.4 Shared-memory tiling — `02_stencil_shared_memory_tiling.cu`

Applies the same shared-memory tiling used for tiled convolution (Ch. 7,
Fig. 7.12): the thread block matches the *input* tile size
(`IN_TILE_DIM^3`), every thread loads one element into `in_s`, and only
interior threads (inside both the valid grid interior and the tile's own
halo layer) compute and write an output point. Unlike convolution, the
7-point stencil's input tile never includes corner grid points (§8.4, Fig.
8.7), and its compute phase never reads a ghost-cell shared-memory slot, so
—unlike Ch. 7's kernel — out-of-range loads are simply skipped rather than
zero-filled.

§8.3/§8.4 both stress that the 1024-thread block-size cap makes cubic 3D
tiles small in practice: `t=8` (512 threads) is the practical limit used
here, matching the book's own worked example. At `t=8`, ~58% of the input
tile is halo (vs. ~12% for a comparable 2D `32x32` convolution tile), so
arithmetic intensity only reaches 0.96 FLOP/B — better than file 01's 0.41,
but still well under the 1.625 FLOP/B ideal. This shortfall is exactly what
motivates thread coarsening in §8.5.

Tested with grid sizes that are and aren't multiples of `OUT_TILE_DIM` (6),
exercising both partial output tiles and ghost-cell handling on every face.

## §8.5 Thread coarsening — `03_stencil_thread_coarsening.cu`

A cubic block of side `t` needs `t^3` threads, capping `t` at 8. §8.5's fix:
coarsen each thread's responsibility from one output point to a whole
*column* of output points along z, so the thread block only needs `t^2`
threads regardless of how many z-planes get swept — the missing z-extent of
the thread grid is replaced by a software loop inside the kernel (Fig.
8.10). This lets `t` grow to 32 (`32x32` = 1024 threads, the hardware max),
raising arithmetic intensity to 1.52 FLOP/B (§8.5) — much closer to the
1.625 FLOP/B ideal than file 02's 0.96.

At any instant, only 3 z-planes of the input tile are needed to compute the
plane currently being swept: previous (z-1 neighbor), current (4 in-plane
neighbors + center), and next (z+1 neighbor). Rather than stage the whole
`t^3` cubic tile in shared memory, this kernel keeps only those 3 `t^2`
planes (`inPrev_s`/`inCurr_s`/`inNext_s`) live, sliding the window forward
by one plane after each iteration — `3*t^2` elements (12 KB at `t=32`,
matching §8.5's own figure) instead of `t^3`.

This is a genuinely different technique from file 02's tiling, not a
renamed copy: file 02 launches one thread per 3D grid point (one block per
small 3D tile); this file launches one thread per 2D (y,x) column and loops
over the z output planes in software — fewer threads and fewer blocks
launched per output point processed, with the z-sweep loop taking over the
role the thread grid's missing z-dimension used to play.

Tested with grid sizes that are and aren't multiples of `OUT_TILE_DIM` (30),
exercising both partial coarsening ranges and ghost-cell handling on every
face.

## §8.6 Register tiling — `04_stencil_register_tiling.cu`

File 03 keeps all 3 active z-planes in shared memory, but `inPrev_s[y][x]`
and `inNext_s[y][x]` are each read by exactly **one** thread — the thread
owning that `(y,x)` column — during its own z-neighbor terms. Only
`inCurr_s` is genuinely shared, since in-plane neighbor terms (`c1..c4`)
need values that belong to *other* threads' columns. §8.6's optimization:
stop broadcasting `inPrev`/`inNext` through shared memory at all — make them
plain per-thread register variables. `inCurr` is kept in *both* a register
(for this thread's own center-point term) and in `inCurr_s` (so neighboring
threads can still read this thread's value for their in-plane terms) — Fig.
8.12's kernel builds directly on file 03's coarsening loop, replacing 2 of
its 7 shared-memory reads per output point (the z-neighbor terms) with
register reads.

This is a genuinely different technique from files 02 and 03, not a
cosmetic rename: it keeps only **one** `t^2` plane in shared memory instead
of three, dropping shared-memory use per block to 1/3 of file 03's (4 KB vs.
12 KB at `t=32`, matching §8.6's own numbers), at the cost of 2 extra
registers per thread. Global memory traffic and total data reuse are
unchanged from file 03 — §8.6 notes register tiling only redistributes
*where* on-chip reuse happens, it does not further reduce DRAM bandwidth
demand.

Tested with the same grid sizes as file 03, for direct comparability.

## §8.3, §8.7-8.8 Notes not given a separate file

- **§8.3 arithmetic intensity.** For an ideal 3D 7-point stencil sweep on an
  `n x n x n` grid where every input point is loaded exactly once, the
  arithmetic intensity approaches `13/8 = 1.625` FLOP/B as `n` grows — much
  lower than convolution's comparable ratio, because stencils reuse far
  fewer of their neighbors' *loaded* values (no diagonal/corner taps) and
  because 3D stencils are the chapter's primary use case (2D convolution's
  neighborhoods are denser and its arithmetic intensity scales with filter
  area, not just filter radius).
- **§8.7 Summary / §8.8 Exercises** are conceptual recap and computed
  problems with no new kernel listing; the chapter's core message — stencils
  resemble convolution but their 3D-grid, sparse-neighborhood nature
  motivates thread coarsening and register tiling specifically, rather than
  the halo-caching tricks of Ch. 7 — is reflected in files 03/04 above.

Build and run all samples in this chapter:

```sh
make run
```
