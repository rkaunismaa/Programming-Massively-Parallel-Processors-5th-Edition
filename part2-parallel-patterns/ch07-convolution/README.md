# Chapter 7: Convolution and constant memory

Source: *Programming Massively Parallel Processors*, 5th ed., Ch. 7 (pp. 159-181).

| File | Book section | Technique |
|------|-------------|-----------|
| `01_convolution_naive.cu` | §7.2 | `convolution_2D_basic_kernel` (Fig. 7.7): one thread per output pixel, filter `F` in global memory, ghost cells (ghost = 0) handled with a per-tap bounds check |
| `02_convolution_constant_memory.cu` | §7.4 | `convolution_2D_const_mem_kernel` (Fig. 7.9): identical kernel, `F` moved to `__constant__` memory and populated with `cudaMemcpyToSymbol` |
| `03_convolution_tiled_halo.cu` | §7.5 | `convolution_tiled_2D_const_mem_kernel` (Fig. 7.12): shared-memory tiling; the whole input tile *including* its halo is explicitly staged into `__shared__` memory before any output element is computed |
| `04_convolution_tiled_cache_halo.cu` | §7.6 | `convolution_cached_tiled_2D_const_mem_kernel` (Fig. 7.15): shared memory holds only the tile's interior; halo accesses read `N` directly from global memory, relying on the L2 cache |

§7.1 (Background) introduces convolution and its "ghost cell" boundary
handling using a *1D* running example (Figs. 7.1-7.3) purely to build
intuition, then immediately generalizes to 2D for image processing (Fig.
7.4-7.5): "we show code examples for 2D convolution and the reader is
encouraged to adapt these code examples to 1D and 3D as exercises" (§7.2).
Every kernel figure in the chapter (7.7, 7.9, 7.12, 7.15) is 2D, so all four
files here implement 2D convolution, matching the book's actual code. §7.3
(memory bandwidth / arithmetic-intensity analysis) and §7.7-7.8 (summary,
exercises) are conceptual/quantitative discussion with no kernel listing of
their own and are summarized below rather than given a separate file.

All four files share the same 2D convolution definition and mask/radius
convention: for output element `P[row][col]`, and a square filter `F` of
radius `r` (dimension `2r+1`), the weighted sum walks `fRow, fCol` in
`[0, 2r]` with corresponding input coordinates `inRow = row - r + fRow`,
`inCol = col - r + fCol`; any input coordinate outside `[0, width) x [0,
height)` is a ghost cell and contributes 0 (§7.1, Fig. 7.5).

## §7.1-7.2 Basic kernel -- `01_convolution_naive.cu`

Each thread is assigned one output element `P[outRow][outCol]`, using the
same 2D thread-to-element mapping as the Chapter 3 color-to-grayscale
kernel (Fig. 7.6). `F` is passed in as an ordinary global-memory pointer.
The doubly-nested loop over the filter accumulates into a register
(`Pvalue`), and each tap is individually bounds-checked against `width`/
`height` before touching `N`, so ghost cells are simply skipped rather than
read (§7.2). Threads near the four edges of `P` skip a different number of
taps than interior threads, producing some control divergence -- the book
notes this is "modest to insignificant" for the large images and small
filters convolution is normally applied to.

§7.3's arithmetic-intensity analysis of this kernel: each loop iteration
loads 2 values (`F` and `N`, 8 B) from global memory for 2 FLOPs, giving
0.25 FLOP/B regardless of filter size -- deep in memory-bound territory.

Tested with a mix of filter radii (1-3) and image sizes that are and aren't
multiples of the 16x16 thread block, so ghost-cell handling on all four
edges is actually exercised.

## §7.4 Constant memory -- `02_convolution_constant_memory.cu`

Identical kernel body to file 01, with `F` moved from a pointer parameter
to a `__constant__` global array sized by the compile-time `FILTER_RADIUS`
(2 here) and uploaded once per test case via `cudaMemcpyToSymbol`. §7.4
gives three reasons `F` suits constant memory: it's small, it's read-only
during kernel execution, and every thread reads it in the same order --
so the specialized constant cache can serve nearly all `F` accesses with no
DRAM traffic. §7.3/§7.4's arithmetic-intensity analysis: with `F` no longer
counted as a DRAM access, each loop iteration is now 1 load (4 B) for 2
FLOPs, i.e. 0.5 FLOP/B -- double the basic kernel's.

## §7.5 Tiled convolution with halo cells -- `03_convolution_tiled_halo.cu`

Applies shared-memory tiling as in Chapter 5's tiled matmul, but
convolution's *input* tile is larger than its *output* tile by
`FILTER_RADIUS` cells on every side (the halo, Fig. 7.11). This file uses
the book's first thread organization (§7.5): the block is sized to match
the *input* tile (`IN_TILE_DIM = 32`), so each thread loads exactly one
element into shared memory `N_s`, with out-of-range loads zero-filled as
ghost cells (Fig. 7.12, lines 09-15) before a `__syncthreads()`. Because the
block is bigger than the output tile, only the interior
`FILTER_RADIUS .. blockDim-1-FILTER_RADIUS` threads are "active" during the
compute phase (Fig. 7.13) and write a `P` element, reading their
filter-sized patch out of `N_s`. With `FILTER_RADIUS = 2`, `OUT_TILE_DIM =
32 - 2*2 = 28` -- deliberately not a power of two, which §7.5 calls out as
one of this design's inefficiencies (along with wasted threads that only
load and never compute), addressed by file 04.

Tested with image sizes that are and aren't multiples of `OUT_TILE_DIM`
(28), so both partial output tiles and ghost cells at the array edges are
exercised.

## §7.6 Tiled convolution using caches for halo cells -- `04_convolution_tiled_cache_halo.cu`

A block's halo cells are exactly the *interior* elements of its neighbors'
input tiles, so by the time a block needs them there's a good chance
they're already in the L2 cache from a neighbor's own loads. This kernel
therefore loads *only* the interior tile into shared memory `N_s` (no
halo), so input tile, output tile, and block are all the same size --
`TILE_DIM = 32`, a power of two, which §7.6 highlights as a "subtle
advantage" over file 03's kernel. During the compute phase, each filter tap
is checked against the block's own tile bounds (Fig. 7.15, lines 17-20): if
the needed `N` element is inside the tile it's read from `N_s`; otherwise
it's a halo access, and it's read directly from `N` in global memory
(served from L2 in the common case) unless it's also a true ghost cell
(lines 24-27), in which case it's skipped as 0 -- the same ghost-cell test
used throughout this chapter.

Tested with image sizes that are and aren't multiples of `TILE_DIM` (32),
so both partial tiles and ghost cells at the array edges are exercised.

## §7.3, §7.7 Notes not given a separate file

- **§7.3 arithmetic intensity.** For an `n x n` image and `m x m` filter
  (`m = 2r+1`), convolution performs `2n^2m^2` FLOP while accessing
  approximately `8n^2` B (ignoring the comparatively tiny filter), for an
  ideal arithmetic intensity of `(1/4)m^2` FLOP/B -- e.g. 2.25 FLOP/B for a
  3x3 filter (memory-bound) vs. 30.25 FLOP/B for an 11x11 filter
  (compute-bound on an H100). The tiled kernel of file 03 attains an
  intensity of `(1/4)m^2 * 1/(2*(1+(m-1)/t)+(m-1)^2/t^2)` for output tile
  dimension `t`, approaching the ideal as `t` grows.
- **§7.7 Summary** notes that stencil algorithms in PDE solvers (Chapter 8)
  are a special case of convolution, and that convolutional neural networks
  (Chapter 20) build directly on this chapter's techniques.

Build and run all samples in this chapter:

```sh
make run
```
