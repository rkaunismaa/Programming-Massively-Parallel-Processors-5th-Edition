# Chapter 8 — Stencil Computation

Section-by-section summary of the book content. Page numbers are printed
book pages (Chapter 8 runs pp. 183–199).

## 8.1 Background (p. 184)

Using computers to solve continuous functions/equations numerically
first requires **discretization**: representing a continuous function
on a **structured grid** of evenly-spaced sample points (Fig. 8.1 —
`sin(x)` sampled at 7 points spaced `π/6` apart). Structured grids
(identical parallelotopes — segments in 1D, rectangles in 2D, bricks in
3D) are the book's focus, as they make derivatives convenient to
express as finite differences (unstructured grids, used by
finite-element/finite-volume methods, are more complex and out of
scope). Grid spacing trades accuracy for cost: finer spacing improves
fidelity but increases both storage and compute; floating-point
precision (double vs. single vs. half) trades the same way, and lower
precision also reduces the memory bandwidth and register/shared-memory
footprint needed for tiling (tying back to Chapter 7).

A **stencil** is a geometric pattern of fixed weights applied at each
grid point, derived from a numerical approximation routine — most
commonly a finite-difference approximation of a derivative, since
partial differential equations (PDEs) are expressed in terms of
derivatives. Worked example: the classic central-difference
approximation `f'(x) ≈ (f(x+h) - f(x-h))/(2h)` turns into, for a
discretized array `F` with spacing `h`, `FD[i] = (F[i+1]-F[i-1])/(2h) =
(-1/2h)*F[i-1] + (1/2h)*F[i+1]` — a **1D 3-point stencil** (Fig. 8.2a)
with weights `[-1/2h, 0, 1/2h]` on grid points `[i-1, i, i+1]`.
Approximating higher-order derivatives needs more neighboring points on
each side; in general, approximating up to the `n`-th derivative needs
`n` points on each side of the center, giving a `(2n+1)`-point stencil
— the count of points on *each* side is called the stencil's **order**
(Fig. 8.2's 3-/5-/7-point 1D stencils have orders 1, 2, 3
respectively). The same idea generalizes directly to 2D (Fig. 8.3a–b:
5-point and 9-point stencils, orders 1 and 2, used when a PDE involves
only pure `∂/∂x` and `∂/∂y` terms, not mixed `∂²/∂x∂y`) and 3D (Fig.
8.3c–d: 7-point and 13-point). Fig. 8.4 illustrates applying a 5-point
stencil across an entire grid, stored as a 2D array — processing every
grid point with the stencil is called a **stencil sweep**, the
chapter's central computation pattern.

## 8.2 Parallel stencil — a basic kernel (p. 187)

Since output grid points are independent of each other (within one
sweep), stencil sweep parallelizes exactly like convolution: one
thread per output grid point. The chapter adds one simplifying
assumption used throughout: **boundary grid points hold fixed boundary
conditions and are not recomputed** between sweeps (Fig. 8.5) — a
reasonable assumption since stencils are typically used to solve PDEs
with boundary conditions, and it means only strictly-interior points
need output values computed at all (no ghost-cell handling is needed
for *output* writes, only for *input* reads near those boundaries).

Fig. 8.6's `stencil_kernel` targets a **3D grid with a 7-point
stencil** (the book's primary real-world case — most practical PDE
stencil applications are 3D), using the familiar
`blockIdx*blockDim+threadIdx` mapping for all three dimensions
(`i`,`j`,`k`), bounds-checked to the interior (`1..N-2` in each
dimension), and computing `out[...] = c0*in[i,j,k] + c1..c6 *` each of
the 6 face-neighbors. The `c0..c6` coefficients may be hard-coded,
passed as parameters, or placed in constant memory, depending on the
specific PDE being solved and how much flexibility is needed.

## 8.3 Memory bandwidth considerations (p. 189)

Applying the same Chapter 5/7 roofline analysis: for an `n×n×n` grid,
`(n-2)³` interior points are computed, each needing 13 FLOP (7
multiplies, 6 adds) — `13·(n-2)³` FLOP total. Input traffic is every
grid point except edges/corners (`n³ - 12n + 16` values); output
traffic is just the `(n-2)³` interior points; all values are 4 B. The
**ideal** arithmetic intensity works out to `13·(n-2)³ /
(4·(n³-12n+16+(n-2)³))`, which for large `n` approaches **13/8 =
1.625 FLOP/B** — very low, making the 3D 7-point stencil inherently
memory-bound on modern GPUs (much like small convolution filters in
Chapter 7).

The *actual* intensity of the basic kernel (Fig. 8.6) is lower still:
each thread loads 7 distinct 4 B input values and stores 1, for
`13/((7+1)*4) = 0.41 FLOP/B` — well below the 1.625 FLOP/B ideal,
because (as with the basic convolution kernel) every access goes
straight to global memory with no reuse across threads. Shared memory
tiling (§8.4) is the fix, as in Chapter 7.

## 8.4 Shared-memory tiling for stencil sweep (p. 189)

Tiling for stencils closely mirrors tiling for convolution (Fig. 8.7
vs. Fig. 7.11), with one structurally important difference: a stencil's
input tile **does not include the corner grid points** that a
same-radius convolution filter's input tile would need (the stencils in
Fig. 8.3 only touch face/axis neighbors, never diagonal ones) — so
stencils inherently reuse less input data per output element than a
same-sized convolution filter would. For a 2D 5-point stencil, the
*ideal* arithmetic intensity is only 1.125 FLOP/B, versus 2.25 FLOP/B
for a same-footprint 3×3 convolution; the gap widens with stencil
order (2.125 vs. 6.25 FLOP/B at order 2; 3.125 vs. 12.25 FLOP/B at
order 3) and widens dramatically moving to 3D (a 3rd-order 3D stencil:
4.625 FLOP/B, vs. 85.75 FLOP/B for an equivalent 3D 7×7×7 convolution).
This structurally lower reuse is exactly what later motivates thread
coarsening (§8.5) specifically for stencils.

Fig. 8.8's `stencil_kernel` applies the same shared-memory-tiling
strategy as Chapter 7's tiled convolution kernel (input tile size =
block size, output tile smaller, outer `FILTER_RADIUS`-equivalent
threads deactivated during computation — here the deactivation radius
is 1, since a 7-point stencil has order/radius 1), loading one grid
point per thread into `in_s` (bounds-checked against both the lower
bound `>=0` for ghost cells and the upper bound `<N` for the grid's
true edge), then barrier-syncing before the interior threads compute
their output using only `in_s`.

Its arithmetic intensity, for cubic input tiles of width `t` (output
tiles `t-2`): `13·(t-2)³ / (4·t³+4·(t-2)³) = 13/8 · 1/(½·(t³/(t-2)³) +
½)` FLOP/B — approaching the 1.625 FLOP/B ideal as `t` grows. But `t`
is hardware-limited: the 1024-thread/block cap means a cubic block can
be at most 8×8×8 (`t=8`), giving only **0.96 FLOP/B** — far short of
ideal, since at small `t` a large fraction of the input tile is
low-reuse **halo** rather than interior data (illustrated numerically:
a 2D 32×32 convolution input tile is only ~12% halo, but an 8×8×8 3D
stencil input tile is ~58% halo). A cubic 8×8×8 block also *coalesces
poorly*: a warp (32 threads) spans 4 separate rows of 8 elements each,
touching 4 distant memory locations per load — fixable by using
non-cubic block shapes (e.g. 32×8×4) so a warp's 32 threads stay within
one contiguous row, though this doesn't fix the underlying small-`t`
arithmetic-intensity problem. Both limitations motivate thread
coarsening.

## 8.5 Thread coarsening (p. 193)

**Thread coarsening** overcomes the cubic block-size limit by having
each thread compute a whole *column* of output grid points (coarsening
along `z`) rather than a single point, so the thread block only needs
`t²` threads instead of `t³` — letting `t` grow much larger (e.g.
`t=32` → 1024 threads, vs. a cubic block's `t=8` cap) while needing
only `3·t²` shared-memory elements resident at once (three `z`-planes:
`inPrev_s`, `inCurr_s`, `inNext_s`), not all `t³` (Fig. 8.9).

Fig. 8.10's kernel: each thread is first assigned one (x,y) position
and iterates over `z` from `iStart` to `iStart+OUT_TILE_DIM`. Each
iteration: cooperatively load the next `z`-plane into `inNext_s`
(bounds-checked), barrier-sync, compute the current output plane using
4 same-plane neighbors from `inCurr_s` plus the single `z`-neighbors
from `inPrev_s`/`inNext_s`, barrier-sync again, then **rotate** the
three plane buffers (`inPrev_s ← inCurr_s ← inNext_s`) for the next
iteration (Fig. 8.11). For `t=32`, this reaches **1.52 FLOP/B** —
dramatically closer to the 1.625 FLOP/B ideal than the 0.96 FLOP/B of
the uncoarsened 8×8×8 tiled kernel — at a shared-memory cost of just
`3·32²·4 = 12 KB` per block, a reasonable level.

## 8.6 Register tiling (p. 196)

Observes that, for stencils touching only axis-aligned neighbors (true
of every stencil in Fig. 8.3), the `inPrev_s`/`inNext_s` z-neighbor
values in the coarsening kernel are each used by only **one** thread —
the same thread that loaded them — so they never actually need to live
in *shared* memory; only `inCurr_s` (whose x-y neighbors are read by
*multiple* different threads) genuinely needs to be shared. Moving
`inPrev`/`inNext` into plain per-thread **registers** instead
(**register tiling**, reusing the Chapter 5/6 concept) is a direct
modification of Fig. 8.10's kernel (Fig. 8.12): `inPrev` and `inNext`
become scalar register variables loaded/rotated the same way as
before, while `inCurr_s` stays a shared-memory array as before (and is
still updated each iteration, since other threads need to read a
thread's own `inCurr` value).

This reduces shared-memory consumption to **one third** of the
coarsening-only kernel (only one plane resident in shared memory
instead of three), at the cost of 3 extra registers per thread (3072
more registers per 32×32 block) — a direct shared-memory-for-register
tradeoff, not a reduction in overall data reuse or global memory
traffic (the total DRAM accesses, and hence arithmetic intensity, are
unchanged from the coarsening-only kernel — the data is simply spread
across registers and shared memory rather than shared memory alone).
If register pressure from a higher-order stencil becomes a problem,
some planes can be moved back into shared memory — a tradeoff to make
case by case.

## 8.7 Summary (p. 198)

Stencil sweep is, in essence, convolution with a special (sparse,
axis-aligned-only) filter pattern — but because stencils arise from
discretizing and numerically approximating derivatives, they have two
distinguishing characteristics that motivate techniques beyond plain
Chapter 7 tiling: (1) stencils are typically applied to **3D** grids
(unlike convolution's typical 2D-image use case), which, combined with
hardware thread-block-size limits, motivates **thread coarsening** to
achieve usably large tile sizes and arithmetic intensity; (2) a
stencil's axis-aligned-only access pattern (no diagonal/corner
neighbors) enables **register tiling** of the coarsened dimension's
neighbor values, further improving data-access throughput and easing
shared-memory pressure.

## 8.8 Exercises (p. 198)

Two multi-part problems: (1) for a 120×120×120 3D grid (with boundary
cells), computing the number of output grid points per sweep and the
number of thread blocks needed under the basic kernel (Fig. 8.6, 8×8×8
blocks), the shared-memory-tiled kernel (Fig. 8.8, 8×8×8 blocks), and
the coarsened kernel (Fig. 8.10, 32×32 blocks); (2) for a 7-point 3D
stencil with 32×32 blocks and a coarsening factor of 16, computing the
input-tile element count, output-tile element count, arithmetic
intensity, and per-block shared-memory bytes needed with and without
register tiling (Figs. 8.10 vs. 8.12). Not implemented in this repo's
samples (end-of-chapter exercises are out of scope — see the repo root
README).
