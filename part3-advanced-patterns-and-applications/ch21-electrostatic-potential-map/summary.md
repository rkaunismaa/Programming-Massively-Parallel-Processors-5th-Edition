# Chapter 21 — Electrostatic Potential Map

Section-by-section summary of the book content. Page numbers are printed
book pages (Chapter 21 runs pp. 513–528).

This chapter uses a real molecular-dynamics application (based on VMD —
Visual Molecular Dynamics) to illustrate memory-coalescing and
throughput optimization on **regular-grid** data, via a progression of
kernel versions, each improving on the last. Several techniques overlap
with Chapter 6's checklist; others are specific to this pattern:
**gather-vs-scatter kernel design**, **systematic reuse of computational
results**, and **fast boundary-condition (cutoff) checking**.

## 21.1 Background (p. 513)

VMD is popular software for displaying, animating, and analyzing
bio-molecular systems — an important "computational microscope" for
observing life forms too small for traditional microscopy, and a
general-purpose tool for visualizing other large datasets (sequencing,
quantum chemistry, volumetric data). Multiple VMD computations have
been CUDA-accelerated, including the chapter's focus: **electrostatic
potential map** calculation — computing the potential at every point
of a spatial grid around a molecular structure (Fig. 21.1), used both
for placing ions into a structure in preparation for simulation and
for time-averaged field analysis during/after simulation.

The chapter's method is **Direct Coulomb Summation (DCS)** — a highly
accurate method well-suited to GPUs: each grid point's potential is
the sum, over **all** atoms in the system, of each atom's charge
divided by its distance to that grid point (Fig. 21.2). Since this
sum must be computed for every grid point against every atom, total
work scales with (atom count) × (grid point count) — for realistic
molecular systems, both grow with system size, so this product grows
as roughly the **square** of system volume (addressed in §21.5).

## 21.2 Scatter vs. gather in kernel design (p. 515)

Fig. 21.3's unoptimized sequential C `cenergy()` processes one 2D grid
slice with three nested loops (`y`, `x`, then innermost over all
atoms), recomputing each atom's `dx`/`dy`/`dz` distance components from
scratch for every grid point. Fig. 21.4's **optimized** C version
**interchanges** the loop nest — atoms become the outermost loop,
grid points the two inner loops — which is valid since all iterations
are mutually independent. This interchange enables two big wins: the
`z`-distance component is now identical for the whole 2D slice and
computed **once per atom** (not per grid point), and the `y`-distance
component is identical for a whole row and computed once per row —
eliminating the vast majority of redundant distance-component
recomputation that the unoptimized version paid for every (atom, grid
point) pair.

Parallelizing the **optimized** C version directly gives Fig. 21.5's
**scatter** kernel: one thread per atom, each thread **scatters** its
own atom's contribution out to every grid point it affects — but since
multiple atoms (threads) write to the *same* grid point, this requires
**atomic** operations on every grid-point update, which "significantly
reduces the speed of parallel execution."

The alternative is a **gather** kernel (Fig. 21.6): one thread per grid
point, each thread **gathers** the summed contribution of all atoms
into its own uniquely-owned grid point — no atomics needed, since each
thread only ever writes its own location. But achieving this requires
parallelizing the loop ordering of the **unoptimized** code (Fig. 21.3),
not the optimized one, since the gather structure needs atoms as the
*innermost* loop. This illustrates "a frequently experienced dilemma in
parallelizing applications": the *most-optimized sequential* code is
not always the most parallelization-friendly code, and the resulting
parallel kernel can run a visibly less-optimized *inner loop* per
thread, even though it avoids atomics entirely — a tradeoff the chapter
returns to and ultimately resolves via thread coarsening (§21.3).

Despite its less-optimized inner loop, Fig. 21.6's gather kernel
already performs well: each thread does 9 FLOPs per 4 `atoms[]`
elements accessed, and since all threads **broadcast-read the same**
`atoms[]` array (held in **constant memory**), the hardware constant
cache serves almost all of these accesses, eliminating the vast
majority of DRAM traffic — so global memory bandwidth is not yet the
bottleneck for this kernel.

## 21.3 Thread coarsening (p. 519)

Even with constant caching, Fig. 21.6 still executes **4 constant-
memory-access instructions per 9 FLOPs** — each access instruction
consumes hardware issue slots and energy that could otherwise go toward
more FLOPs. Since all grid points along one row share the same `y`
coordinate, the `dy`/`dz`-derived distance terms the kernel recomputes
per grid point are actually **identical across several grid points in
the same row** — a redundancy the single-grid-point-per-thread version
pays for repeatedly. **Thread coarsening** (Chapter 6) fixes this: Fig.
21.8 has each thread compute **4 grid points** in the same row at once.
For each atom, `dy` and `dysqdzsq` (`dy² + dz²`) are computed **once**
and reused for all 4 grid points (stored in registers); the atom's
charge is likewise loaded from constant memory once and reused four
times; only the `x`-distance component (`dx0`..`dx3`) differs per grid
point, computed cheaply as offsets from a shared base.

Net effect per atom, per 4 grid points: constant-memory accesses drop
from 16 (Fig. 21.6) to 4 — a 4× reduction — and total floating-point
operations drop from 48 to 24 (also roughly halved), a "sizable
reduction" the book expects to translate into sizable execution-time
and energy savings. The cost is more registers used per thread, but
since this stays within the device's limits, occupancy is unaffected
in this case.

## 21.4 Memory coalescing (p. 521)

Profiling Fig. 21.8 reveals **uncoalesced** global memory **writes**:
each thread's 4 grid points are **adjacent** to each other (`i`,
`i+1`, `i+2`, `i+3`), but **different threads** in a warp are *not*
adjacent to each other in the same way — each thread "jumps" past its
own 4-point block to the next, spreading the 32 threads' actual
write addresses across a much wider range than a single coalesced
transaction can cover.

Fix (Fig. 21.9): re-partition which 4 grid points each thread owns, so
that adjacent *threads* get adjacent individual grid points (not
blocks of 4), while each individual thread's 4 assigned points are
instead spaced `blockDim.x` elements apart — assign the first
`blockDim.x` consecutive grid points to the first `blockDim.x`
threads (one each), then the next `blockDim.x` consecutive points to
the same threads (second assignment each), repeating until every
thread has its 4 points. Fig. 21.10 implements this: `dx0`..`dx3` are
now spaced `blockDim.x*gridspacing` apart (not `gridspacing`), and the
four `energygrid[]` write indices are similarly spaced `blockDim.x`
apart — so that within any single one of the four write statements,
consecutive threads write consecutive addresses, **fully coalescing**
every write. (The same coarsening factor could alternatively use
**vector stores** on the original, un-reassigned thread-to-grid-point
mapping — left as an exercise.)

## 21.5 Cutoff binning for data size scalability (p. 523)

Different algorithms for the same problem can trade off computation
steps, exposed parallelism, numerical stability, and memory bandwidth
— rarely is one option strictly best on all four axes, so a parallel
programmer typically picks the best compromise for the target
hardware. Here, **cutoff summation** is introduced as a more
*aggressive* algorithmic variation that trades a small amount of
accuracy for a large efficiency win, based on the physical observation
that a grid point's contribution from a *distant* atom is tiny
(inversely proportional to distance) and can be handled by a cheaper,
implicit approximation instead (Fig. 21.11): only atoms within a fixed
cutoff radius are summed directly. This changes DCS's complexity from
proportional to **system volume squared** (atoms × grid points, both
scaling with volume) to proportional to **volume directly** — the
difference between "excessively long even on massively parallel
hardware" at large scale and genuinely scalable.

A naive sequential cutoff implementation handles one atom at a time
(easy, since grid points are array-indexable by coordinate) — but this
is again **atom-centric**, inheriting the same atomic-operation
problem that made the scatter kernel (§21.2) slow. The chapter instead
adapts the **grid-centric** gather kernel (Fig. 21.10) into a
**cutoff binning** algorithm (citing Rodrigues et al.): atoms are first
sorted into spatial **bins** (a multi-dimensional array indexed by
`x`,`y`,`z` bin coordinate, each bin holding a list of atoms whose
coordinates fall inside it). A grid point's **neighborhood** is the set
of bins that could contain any atom within the cutoff radius (Fig.
21.12) — conservatively approximated per *block* (not per individual
grid point) by a "super circle" centered on the bin containing the
block's grid points, with radius = cutoff distance + half the bin's
diagonal, guaranteeing it covers every possible per-point circle a
block's threads could need (Figs. 21.13–21.14). This neighborhood-bin
list is precomputed on the host as a small table of **relative bin
offsets** (e.g. 9 fully-covered + 12 partially-covered = 21 bins in the
book's small example) and supplied to the kernel (e.g. via constant
memory).

At kernel execution, all threads in a block iterate through this same
neighborhood-bin list, collaboratively loading each neighborhood bin's
atoms into **shared memory**, after which each thread individually
checks (for each shared atom) whether it actually falls within *its
own* grid point's exact cutoff radius — since the per-block
neighborhood is a conservative over-approximation, this per-thread
check is necessary and can cause some **control divergence** within a
warp. Because atoms are typically much sparser near a given point than
the whole-system atom count, this reduces the atoms each thread must
examine to a much smaller, bounded subset — the source of the
algorithm's scalability gain — though it also makes **constant memory**
a poor fit for atom storage (different blocks now need different,
large sets of atoms — no longer a small array reusable identically by
every thread), motivating the shared-memory staging approach instead.

A **subtle binning issue**: real atom distributions are not uniform, so
bins end up with widely varying atom counts. Padding every bin to a
fixed size with dummy (zero-charge) atoms (needed to preserve coalesced
access/transfer patterns) wastes memory/bandwidth and execution time on
sparsely-populated bins. The practical compromise: size bins to
comfortably cover the *vast majority* of cases, and maintain a separate
**overflow list** for atoms that don't fit their home bin; after the
GPU kernel completes, the **host** sequentially processes the (small,
typically <3% of atoms) overflow list to patch in their missing
contribution — and since this host-side cleanup can run **concurrently**
with the *next* kernel launch's device-side work, its latency can be
largely or fully hidden.

## 21.6 Summary (p. 527)

The chapter walked through a series of design decisions and tradeoffs
in parallelizing electrostatic-potential-energy calculation on a
regular grid: parallelizing a highly-optimized sequential DCS
implementation directly leads to a slow **scatter** kernel needing
heavy atomics; parallelizing a *less*-optimized sequential version
instead gives a much faster **gather** kernel; **thread coarsening**
then reclaims most of the redundant-computation savings the more
optimized sequential version had offered; and careful choice of
*which* grid points get folded into each thread yields **fully
coalesced** memory write patterns. DCS itself, however, is not a
scalable method (its operation count grows quadratically with system
volume); **cutoff summation** with **binning** trades a small accuracy
loss for linear-in-volume complexity, retaining a high degree of
parallelism while making the method viable for large, realistic
biological systems.

## 21.7 Exercises (p. 528)

Five problems: complete the host code (grid/block configuration, kernel
call) for the gather kernel of Fig. 21.6 (Q1); compare the number of
operations (memory loads, FP arithmetic, branches) per iteration between
Fig. 21.6 and the coarsening-factor-8 version of Fig. 21.8, accounting
for the 1:8 thread correspondence (Q2); give two potential disadvantages
of increasing per-thread work, as done in §21.3 (Q3); revise the
coarsened kernel (Fig. 21.8) to use vector stores for coalescing instead
of the thread-reassignment approach of §21.4 (Q4); and use Fig. 21.13 to
explain how control divergence can arise when threads in a block share
a neighborhood-bin list (Q5). Not implemented in this repo's samples
(end-of-chapter exercises are out of scope — see the repo root README).
