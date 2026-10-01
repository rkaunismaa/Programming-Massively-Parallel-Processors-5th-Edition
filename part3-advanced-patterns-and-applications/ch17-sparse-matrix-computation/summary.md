# Chapter 17 — Sparse Matrix Computation

Section-by-section summary of the book content. Page numbers are printed
book pages (Chapter 17 runs pp. 401–423).

## 17.1 Background (p. 401)

A **sparse matrix** has mostly-zero elements (Fig. 17.1); storing and
processing those zeros wastes memory capacity, bandwidth, time, and
energy. Sparse matrices commonly represent sparsely-coupled linear
systems (`A×X+Y=0`): each matrix row is one equation, and most
real-world equations involve only a handful of the total variables.
Directly inverting `A` to solve for `X` is usually impractical — large
sparse systems can overwhelm exact methods, and inversion tends to
introduce many new non-zero **"fill-ins"**, often making the inverted
matrix far less sparse (and far larger) than the original. Instead,
large sparse linear systems are commonly solved **iteratively** (e.g.
**Conjugate Gradient** methods, applicable when `A` is positive-
definite): guess `X`, evaluate `A×X+Y`, and refine the guess based on
how far the result is from zero — closely related to the iterative PDE
solvers from Chapter 8.

The dominant cost in these iterative methods is **SpMV** (Sparse
Matrix-Vector multiplication and accumulation, `A×X+Y`, Fig. 17.2) — `A`
is sparse, but `X`/`Y` are typically **dense** vectors. The chapter uses
SpMV throughout to compare sparse storage formats. All such formats
remove zero elements via some **compaction** technique, at the cost of
introducing irregularity — which in turn can hurt memory bandwidth
utilization, cause control divergence, and create load imbalance.
Design considerations used to evaluate every format in this chapter:
**space efficiency** (compaction level achieved), **flexibility** (ease
of adding/removing non-zeros), **accessibility** (what data a given
non-zero's neighborhood lets you reach easily — e.g. "all non-zeros in
this row"), **memory access efficiency** (coalescing), and **load
balance** (even work distribution across threads).

## 17.2 A simple SpMV kernel with the COO format (p. 404)

**Coordinate (COO)** format stores non-zero values in a flat `value`
array, with parallel `rowIdx`/`colIdx` arrays giving each value's
original row/column (Fig. 17.3) — completely eliminates zero storage,
at the cost of two extra index arrays (for very sparse matrices, e.g.
1% non-zero, this overhead is small relative to the space saved).

Parallelization: one thread per **non-zero element** (not per row) —
each thread reads its own `rowIdx`/`colIdx`/`value` entry, multiplies by
the corresponding `x` vector element, and **atomically** accumulates
into `y[row]` (Figs. 17.4–17.5) — an atomic is required since multiple
threads (non-zeros from the same row) may update the same `y` element.
Design-consideration review: COO is highly **flexible** (new non-zeros
can simply be appended to all three arrays, with no ordering
requirement — Fig. 17.6 shows COO's elements can be freely reordered
without losing information, since each entry carries its own
coordinates); it gives easy **accessibility** to a non-zero's row/column
given its index, but *not* easy access to "all non-zeros in row R"
without a scan; its physical layout gives fully **coalesced** memory
access; and because every thread handles exactly one non-zero, there's
essentially **no load imbalance**. Its main drawback is the **atomic
operations** needed for accumulation — both for their latency/
throughput cost (Chapter 9) and because atomics make the accumulation
order **non-deterministic**, which can cause floating-point numerical
instability (sidebar reference to Appendix A). Avoiding atomics would
require all of a row's non-zeros to be handled by the *same* thread —
exactly the accessibility COO lacks.

## 17.3 Grouping row non-zeros with the CSR format (p. 407)

**Compressed Sparse Row (CSR)** groups non-zeros **by row** in the
`value`/`colIdx` arrays (Fig. 17.7), replacing COO's per-element
`rowIdx` array with a much smaller `rowPtrs` array (one entry per row,
plus one sentinel) giving each row's starting offset — `rowPtrs[r+1]`
conveniently also marks row `r`'s end, including an extra sentinel
entry for the last row.

Parallelization: one thread per **row** (Figs. 17.8–17.9) — each thread
loops `rowPtrs[row] .. rowPtrs[row+1]-1`, accumulating into a private
`sum`, with **no atomic needed** since each row (and hence each `y`
output) is now owned by exactly one thread. Versus COO: CSR is **more
space-efficient** (two full-size arrays plus one `numRows+1`-sized
array, vs. COO's three full-size arrays); **less flexible** (inserting
a non-zero into a row requires shifting all later rows' data and
incrementing their offsets — expensive, vs. COO's simple append);
**better row accessibility** (enables the atomic-free row-per-thread
parallelization) but **no column accessibility** at all; **poor memory
access efficiency** (consecutive threads' rows are unrelated in memory,
so the parallel dot-product loop's accesses are **not coalesced** —
worked example: threads 0–3 access `value[0]`, `value[2]`, `value[5]`,
`value[7]` in the first iteration); and **potentially severe control
divergence**, since adjacent rows can have very different non-zero
counts, directly varying each thread's loop trip count. §17.4–17.7
each address one of these weaknesses.

## 17.4 Improving memory coalescing with the ELL format (p. 410)

**ELL** (named for the ELLPACK package) fixes CSR's coalescing problem
via **padding + transposition**. Starting from grouped-by-row data:
determine the row with the **most** non-zeros, pad every other row with
dummy zero-valued entries to match that length (making the matrix
rectangular), then store the now-rectangular `value`/`colIdx` arrays in
**column-major** order (Fig. 17.10) — equivalent to transposing a
row-major rectangular array. Consequently `rowPtrs` becomes unnecessary:
row `r`'s element `t` is simply at linear index `t*numRows + row`.

Parallelization is otherwise identical to CSR, one thread per row (Fig.
17.11–17.12), but now the dot-product loop's consecutive-thread
accesses land at **consecutive memory addresses** (column-major
layout), giving full coalescing. Tradeoffs versus CSR: ELL is **less
space-efficient** — padding overhead depends heavily on non-zero
distribution, and can be severe if even one row is far longer than the
rest (worked example: a 1000×1000, 1%-dense matrix where one outlier
row has 200 non-zeros while others average 10 would bloat ELL to ~40%
of the uncompressed size, vs. CSR's ~2%, a 20× difference) — motivating
§17.5's fix. ELL is **more flexible** than CSR (a non-zero can replace
an existing padding element in-place, with no shifting, as long as the
row isn't already at the max length) and gives **both** CSR's
row-accessibility *and* COO's element-level accessibility (given a
linear index `i`, the row is simply `i % numRows`), though
parallelizing across non-zero elements (rather than rows) wastes
threads on padding elements, unlike COO. ELL's memory access is
**efficient** (coalesced, as shown), but it still exhibits the **same
control divergence** as CSR — padding doesn't change a row's *actual*
non-zero count, just pads other rows to match the worst case.

## 17.5 Regulating padding with the hybrid ELL-COO format (p. 414)

ELL's space-efficiency and divergence problems are both most severe
when one or a few rows are disproportionately long. Fix: before
converting to ELL, **remove** the excess non-zeros from unusually-long
rows and store *just those* in a separate **COO** representation; use
SpMV/ELL for the (now much more uniform) remaining elements and
SpMV/COO to handle the removed ones — a **hybrid** method (Fig. 17.13).
This directly caps the maximum padding needed (worked example: removing
a few elements from the two longest rows cuts the matrix's max
row-length from 5 to 2, and total padded elements from 22 to 3).
Whether the extra bookkeeping to split COO out of ELL is worth it
"depends": for a one-off SpMV it may not pay off, but in an **iterative
solver** — where the same sparse matrix `A` is used repeatedly across
many iterations, only `x`/`y` changing — the hybrid-format construction
cost is a one-time investment amortized over many SpMV calls.

Versus plain ELL: **better space efficiency** (less padding); **more
flexible** (can append new non-zeros to the COO portion directly, even
when a row has no spare ELL padding slot to replace); **slightly worse
accessibility** (finding all of a row's non-zeros now may require also
searching the separate COO part, if that row overflowed); **memory
access efficiency preserved** (both the ELL part and the COO part are
individually coalesced); and **reduced control divergence** (the
longest, most divergence-causing rows are specifically what got moved
out of the ELL part).

## 17.6 Reducing control divergence with the JDS format (p. 416)

**Jagged Diagonal Storage (JDS)** attacks control divergence directly,
without needing padding at all: **sort rows by non-zero count**
(longest to shortest, or vice versa — the sorted matrix looks roughly
triangular, hence "jagged diagonal"), keeping an auxiliary `row` array
that records each sorted position's *original* row index (maintained
by swapping alongside the data during the sort). After sorting, lay the
data out in column-major order as in ELL (no padding needed, since
rows are already similarly-sized where it matters), with an added
`iterPtr` array marking where each iteration's (= each column's)
non-zero block begins in the flattened arrays, since row lengths
genuinely differ along the sorted order (Fig. 17.14). One thread per
row, iterating via `iterPtr` exactly as CSR/ELL's per-row loop did
(Fig. 17.15) — full implementation left as an exercise. A named
variant: **partition sorted rows into sections**, generate an ELL
representation *per section* (padding only to that section's own max
length) — avoids needing `iterPtr` (replaced by one offset per section)
and sharply reduces total padding versus whole-matrix ELL, since
within a section row lengths are already fairly uniform.

Sorting rows doesn't change the underlying linear system's correctness
— reordering a linear system's **equations** (rows) just needs the
corresponding reordering undone on the **solution** (`row` array lets
this be reversed cheaply); as with the hybrid format, sorting's upfront
cost is worth paying when amortized over an iterative solver's many
SpMV calls. Tradeoffs: JDS is **more space-efficient** than ELL (no
padding, or far less with sectioning); **less flexible** than even CSR
(adding a non-zero can change a row's rank, forcing a re-sort);
**row-accessible** like CSR but, like CSR, **not column-accessible**;
memory access is coalesced like ELL but, because JDS has no padding,
each iteration's starting memory address can't be forced to an
architecturally-aligned boundary the way ELL's fixed-width rows can
(making JDS's coalescing somewhat less efficient in practice); and its
headline strength is **load balance** — sorting rows by length means
threads in the same warp are very likely processing similarly-sized
rows, directly minimizing control divergence.

## 17.7 Column-wise accessibility with the CSC format (p. 418)

All formats so far give row-wise access; some computations instead
need **column-wise** traversal. **Compressed Sparse Column (CSC)** is
CSR with rows and columns swapped: non-zeros grouped **by column** in
`value`/`rowIdx`, with a `colPtrs` array giving each column's starting
offset (Fig. 17.16) — structurally identical to CSR, just transposed.

A one-thread-per-column SpMV kernel is shown for completeness/
comparison (Figs. 17.17–17.18), even though CSC is **not actually a
good fit for SpMV**: it needs **atomics** to accumulate into `y`
(multiple columns can touch the same row — same problem as COO), its
accesses to the **input matrix** are uncoalesced (same problem as CSR,
since consecutive threads/columns are unrelated in memory), and it has
the **same control-divergence** risk as CSR/ELL (column lengths vary).
In short, CSC "combines the worst aspects of COO and CSR" for SpMV. One
genuine advantage: accessing the **input vector** `x` *is* coalesced
and each `x` element is read only once per column (vs. other formats'
repeated/scattered `x` reads) — but this doesn't outweigh CSC's other
costs for SpMV specifically.

CSC's real value is elsewhere: it's the natural fit for **vector-matrix
multiplication** (`x` as the first operand, matrix second — needs
exactly the column-wise traversal CSC provides), and for
**sparse-matrix-sparse-vector multiplication (SpMSpV)** — when `x`
itself is sparse, CSC lets the computation **skip entire columns**
whose corresponding `x` element is zero, since each column's non-zeros
are grouped together.

## 17.8 Summary (p. 421)

Sparse matrix computation is both practically important (many
real-world problems have sparsely-coupled structure) and pedagogically
useful as a clean example of **data-dependent performance behavior** —
unlike most patterns covered earlier, the same kernel's efficiency here
varies with the actual distribution of non-zeros in the input data, not
just its size. Compaction formats remove zero-element storage/compute/
bandwidth waste, at the cost of introduced irregularity; **hybrid
methods** (ELL+COO) and **sorting/partitioning** (JDS) are two general
**regularization** strategies for taming that irregularity — notably,
both *reintroduce* some of what compaction removed (padding, or a
reordering step) specifically to buy back regularity where it matters
most. Sparse representations only pay off when sparsity is high enough
and dense-matrix tiling's advantages (implicit indices, easy tiling) are
outweighed by the space/bandwidth saved from omitting zeros — which is
also why sparse-matrix FLOPS ratings are characteristically much lower
than dense-matrix FLOPS on both CPUs and GPUs. NVIDIA's **cuSPARSE**
library provides production-grade GPU-accelerated routines across these
and other sparse formats.

## 17.9 Exercises (p. 422)

Five problems: represent a given small sparse matrix in COO, CSR, ELL,
and JDS form by hand (Q1); derive, for a general `m×n` matrix with `z`
non-zeros, how many integers each of COO/CSR/ELL/JDS needs to represent
it, noting where the question is under-specified (Q2); implement a
COO→CSR conversion kernel using histogram and prefix-sum primitives
(Q3); implement host code to construct the hybrid ELL-COO format and
perform SpMV with it (launching the ELL kernel on-device, computing the
COO contribution on the host) (Q4); and implement a complete parallel
SpMV kernel for the JDS format (Q5, i.e. §17.6's deferred
implementation). Not implemented in this repo's samples (end-of-chapter
exercises are out of scope — see the repo root README).
