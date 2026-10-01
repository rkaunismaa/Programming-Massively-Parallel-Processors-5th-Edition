# Chapter 7 — Convolution (And Constant Memory)

Section-by-section summary of the book content. Page numbers are printed
book pages (Chapter 7 runs pp. 159–182).

## 7.1 Background (p. 159)

**Convolution** is an array operation where each output element is a
weighted sum of the corresponding input element and a neighborhood of
input elements centered on it. The weights come from a **convolution
filter** array (the book deliberately avoids calling it a "convolution
kernel" to not collide with the unrelated CUDA sense of "kernel").
Convolution applies to data of any dimensionality (1D audio, 2D images,
3D video); the chapter's running examples use 1D and 2D (3D is left as
an exercise, reserved properly for Chapter 8's stencil pattern, which
the book frames as a special case of convolution).

**1D convolution**: given an `n`-element input array `x` and a
`(2r+1)`-element filter `f` (odd-sized so the weighted sum is
symmetric, `r` elements on each side — hence `r` is the filter's
**radius**), `y_i = Σ f_{j+r} * x_{i+j}` for `j = -r..r` (Figs.
7.1–7.2 trace this by hand for a 5-element filter, `r=2`). Near the
array's ends, some needed neighbors don't exist; the typical fix is to
assume a default value (commonly 0) for these missing elements,
referred to as **ghost cells** (Fig. 7.3) — other conventions exist too
(replicate the nearest edge value, or treat the array as circular), and
ghost cells reappear with extra significance once tiling is introduced,
since a block's own tile boundary creates ghost-cell-like edge cases
even away from the true array boundary.

**2D convolution** extends this directly: a 2D filter `f` with
`(2r_y+1) × (2r_x+1)` elements produces `P_{y,x} = ΣΣ
f_{j+r_y,k+r_x} * N_{y+j,x+k}` (Fig. 7.4, worked by hand with a 5×5
filter). 2D boundary conditions (Fig. 7.5) are simply richer versions
of the 1D case — an output element near a corner can be missing
elements from *both* a row boundary and a column boundary
simultaneously.

## 7.2 Parallel convolution — a basic kernel (p. 164)

Since every output element can be computed independently, convolution
parallelizes exactly like the Chapter 3 patterns: one thread per output
element, 2D grid of 2D blocks (Fig. 7.6), reusing the same
`outCol`/`outRow` index calculations as `colorToGrayscaleConversion`.
Fig. 7.7's `convolution_2D_basic_kernel` takes `N` (input), `F`
(filter), `P` (output), the filter radius `r`, and `width`/`height`; a
thread's patch of input elements begins at
`(outRow - r, outCol - r)` — the loop nest (`fRow`, `fCol`, each
`0..2r`) walks the whole patch, and each access is individually
bounds-checked (`inRow >= 0 && inRow < height && inCol >= 0 && inCol <
width`) to skip ghost cells (treated as contributing 0, so skipping the
multiply-add is equivalent to including a 0-valued ghost cell).

This introduces **control divergence**: threads near the four edges of
`P` hit the bounds check differently depending on exactly how close to
an edge they are (a corner thread skips the most iterations, an
interior-adjacent edge thread skips the fewest) — but since convolution
is typically applied to large images where edge pixels are a small
fraction of the total, this divergence is expected to be modest to
insignificant. The more significant performance concern, taken up next,
is memory bandwidth.

## 7.3 Memory bandwidth considerations (p. 167)

Using the Chapter 5 roofline approach: for an `n×n` input and `m×m`
filter (`m = 2r+1`), the total arithmetic work is `2·n²·m²` FLOP (one
multiply + one add per output element per filter tap), and the ideal
memory traffic (loading the filter once and the input once, storing the
output once, ignoring the filter's own negligible size since `m << n`)
is `8·n²` bytes. The ratio gives an ideal arithmetic intensity of
**`¼·m²` FLOP/B** — notably, this depends only on the *filter* size,
not the data size: small filters (e.g. 3×3 → 2.25 FLOP/B) are
inherently memory-bound (far below an H100's ~20 FLOP/B threshold, so
peak compute throughput is architecturally unreachable no matter the
implementation), while large filters (e.g. 11×11 → 30.25 FLOP/B) are
compute-bound and can approach peak throughput with the right
optimizations.

The *actual* arithmetic intensity of the basic kernel (Fig. 7.7) is far
below even this ideal: each loop iteration loads 2 elements (8 B — one
`F`, one `N`) and does 2 FLOP, giving just **0.25 FLOP/B**, independent
of filter size, because the kernel reloads both `F` and `N` from global
memory on every single access rather than reusing cached/on-chip
copies. In practice the real number is somewhat higher thanks to
automatic L1/L2 caching of repeatedly-touched elements, but reliably
closing the gap needs deliberate on-chip memory use: constant memory
(§7.4) for `F`, and shared memory tiling (§7.5) for `N`.

## 7.4 Constant memory and caching (p. 168)

`F` has three properties that make it an ideal fit for **constant
memory**: it's small (filter radius is rarely above 7, i.e. ≤15²
elements in 2D, ≤7³=343 in 3D), it never changes during a kernel's
execution, and every thread reads it in the *same* order
(`F_{0,0}, F_{0,1}, ...` via the nested loop). Declaring
`__constant__ float F[...]` (a global, file-scope declaration) and
copying into it with `cudaMemcpyToSymbol(F, F_h, size)` (instead of
`cudaMalloc`/`cudaMemcpy`) places it in a special region of DRAM backed
by a small, dedicated hardware **constant cache** — not shared with the
general L1/L2 hierarchy (Figs. 7.8–7.10). Because constant memory is
never written by kernels, its cache can skip all the hardware
complexity needed to support writes, letting it be built small (the
whole region is capped at 64 KB) yet extremely effective and
power/area-efficient — especially valuable since, as here, all threads
in a warp typically read the *same* constant address at the same time,
which the constant cache can broadcast with very high effective
bandwidth. (An alternative covered briefly: passing filter weights as
by-value kernel parameters also places them in constant memory
automatically, without an explicit `cudaMemcpyToSymbol` call; hard-coding
compile-time-known weights directly in source is a third option, at the
cost of recompiling for different weights.)

Fig. 7.9's `convolution_2D_const_mem_kernel` is nearly identical to
Fig. 7.7, just reading `F` as a global constant-memory array instead of
a passed pointer parameter. Since the `F` load is now (essentially)
free — served from the constant cache rather than counted as DRAM
traffic — arithmetic intensity doubles to **0.5 FLOP/B** (2 FLOP per
one remaining 4 B `N` load), still far below either the ideal `¼·m²` or
the hardware's compute-bound threshold, since `N` is still reloaded
from global memory on every access. Optimizing `N`'s accesses needs
shared memory tiling, taken up next.

## 7.5 Tiled convolution with halo cells (p. 172)

Defines an **output tile** as the set of output elements one block is
responsible for, and the corresponding **input tile** as the input
elements needed to compute that whole output tile — which, because of
the filter's radius, must extend `r` elements beyond the output tile's
own footprint on every side (Fig. 7.11) to cover **halo cells** (the
input elements shared with neighboring tiles, as distinct from true
out-of-bounds **ghost cells**). This makes input tiles substantially
larger than output tiles — e.g. a toy 4×4 output tile with a 5×5 filter
(`r=2`) needs an 8×8 input tile, 4× the area; for a more realistic
16×16 output tile with the same filter, the input tile is 20×20, only
~1.6× the area — the relative overhead shrinks as the output tile
grows.

The chapter implements one of two possible thread organizations (the
other is left as an exercise): **block dimensions matching the input
tile**, so each thread loads exactly one input element into shared
memory — simple to load, but means some threads (the `FILTER_RADIUS`
outer layers) must sit idle during the output-computation phase, since
the output tile is smaller than the block. Fig. 7.12's
`convolution_tiled_2D_const_mem_kernel`: `IN_TILE_DIM` is the
(constant) input/block tile width, `OUT_TILE_DIM =
IN_TILE_DIM - 2*FILTER_RADIUS`; each thread loads one `N` element (with
a bounds check against the true array edges, writing 0 for ghost
cells) into shared `N_s`, then a `__syncthreads()`, then only the
"interior" threads (`tileCol`/`tileRow` in range, deactivating the
outer `FILTER_RADIUS` layer — illustrated by hand in Fig. 7.13) run the
same patch-summing loop as before, but reading from `N_s` instead of
global memory.

Deriving arithmetic intensity for this kernel (with `t` = output tile
width, `m` = filter width): an internal block does `t²·m²·2` FLOP and
loads `(t+m-1)²·4` bytes (input tile) + stores `t²·4` bytes (output
tile), giving intensity `¼·m² · 1/(½·(1+(m-1)/t)² + ½)` — approaching
the ideal `¼·m²` as `t` grows, since a larger output tile amortizes the
halo-loading overhead over more reused output elements (Fig. 7.14
tabulates this across filter/tile sizes — e.g. a 5×5 filter with a
28×28 output tile, 32×32 input tile, reaches 5.42 FLOP/B against an
ideal of 6.25). Three inefficiencies are named: (1) the halo-loading
threads that get deactivated during computation are wasted compute
capacity — fine for memory-bound kernels (where extra parallelism
during the memory-bound load phase helps), a real cost for compute-
bound ones; (2) output tile dimensions end up non-power-of-two
(`OUT_TILE_DIM = IN_TILE_DIM - 2r`), which can misalign global memory
stores; (3) supporting a larger filter radius shrinks the output tile
(since the input tile is capped by the thread-block/shared-memory
limits), which both lowers arithmetic intensity and worsens the other
two inefficiencies. §7.6 addresses this with a different tradeoff.

## 7.6 Tiled convolution using caches for halo cells (p. 178)

Observes that a block's halo cells are, by definition, **interior**
elements of some *neighboring* block's input tile — so there's a good
chance they're already sitting in the L2 cache by the time this block
needs them, from the neighbor's own recent load. This means halo
accesses can often be served from cache *without* explicitly loading
them into shared memory at all — simply leaving halo-cell reads against
the original global-memory `N` array (relying on L2) rather than
staging them through `N_s`.

Fig. 7.15's `convolution_cached_tiled_2D_const_mem_kernel` only loads
each tile's **interior** elements into shared memory (`N_s`, sized
`TILE_DIM × TILE_DIM`, no separate halo margin) — so input tile and
output tile become the *same* size, and the thread block can be sized
to match both directly, with no idle/deactivated threads during
loading. The output-computation loop now needs to check, for each
patch element, whether it falls inside the tile (read from `N_s`) or
outside it — split into a halo case (still bounds-checked against the
true array edges and read from global `N`, relying on cache) and (by
omission) the ghost-cell case (skipped, same as before). The resulting
kernel can use power-of-two tile/block sizes throughout and avoids the
dead-thread inefficiency of §7.5's approach, at the cost of a somewhat
more complex per-element condition in the compute loop (checking
interior-vs-halo-vs-ghost rather than just interior-vs-ghost).

## 7.7 Summary (p. 179)

Convolution is both an important pattern in its own right (image/video
processing, computer vision) and a general pattern that recurs
elsewhere in the book: stencil computation (Chapter 8) is framed as a
special case of convolution, and much of this chapter's reasoning
carries over directly to convolutional neural networks (Chapter 20).
The chapter progressed through three implementations of increasing
sophistication — a basic DRAM-bandwidth-limited kernel, a version using
constant memory/caching to all but eliminate filter-element traffic,
and two tiled variants using shared memory (and, alternatively, cache
reliance) to cut input-element traffic — each improving arithmetic
intensity, with the tradeoffs made precise via the Chapter 5-style
roofline analysis. The underlying index-calculation and tiling
techniques generalize directly to 3D convolution, just with one more
dimension's worth of loop nesting and index math (left as a homework
exercise rather than worked out in the chapter).

## 7.8 Exercises (p. 180)

Problems include: hand-computing a specific ghost-cell-affected output
value (Q1); hand-computing a full small 1D convolution (Q2); inferring
what several named small 1D filters functionally do, e.g. identity,
shift, or discrete-derivative filters (Q3); counting ghost cells and
multiply operations (with/without treating ghost cells as literal
multiplications) for general 1D (Q4), square 2D (Q5), and rectangular
2D (Q6) convolution given array/filter dimensions. Not implemented in
this repo's samples (end-of-chapter exercises are out of scope — see
the repo root README).
