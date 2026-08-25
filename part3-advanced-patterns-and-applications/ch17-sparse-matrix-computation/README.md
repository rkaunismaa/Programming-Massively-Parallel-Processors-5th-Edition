# Chapter 17: Sparse matrix computation

Source: *Programming Massively Parallel Processors*, 5th ed., Ch. 17 (pp. 401-423).

| File | Book section | Technique |
|------|-------------|-----------|
| `01_spmv_coo.cu` | §17.2 | SpMV/COO: one thread per non-zero, atomic accumulate into `y` (Fig. 17.5) |
| `02_spmv_csr.cu` | §17.3 | SpMV/CSR: one thread per row, dot-product loop, no atomics needed (Fig. 17.9) |
| `03_spmv_ell.cu` | §17.4 | SpMV/ELL: one thread per row, padded column-major storage for coalesced access (Fig. 17.12) |
| `04_spmv_hybrid_ell_coo.cu` | §17.5 | Hybrid ELL+COO: capped ELL for regular rows, COO for overflow from outlier rows |
| `05_spmv_jds.cu` | §17.6 | SpMV/JDS: rows sorted by length, unpadded column-major storage to reduce control divergence |
| `06_spmv_csc.cu` | §17.7 | SpMV/CSC: one thread per column, atomic accumulate, coalesced input-vector reads (Fig. 17.18) |

All six files implement sparse matrix-vector multiplication (SpMV): each
builds a small dense test matrix on the host (either the exact 4x4 example
worked through across Figs. 17.3/17.7/17.10/17.16, or a randomly generated
sparse matrix), converts it to the storage format under test, runs the CUDA
SpMV kernel(s), and checks the result against a CPU dense matvec reference
with `nearlyEqual`, printing a timing line plus `PASS`/`FAIL`.

**The canonical 4x4 matrix.** The book's running example across §17.2-17.7
is never printed as a plain matrix, but its exact values can be reconstructed
from the row/column groupings the text gives for CSR (Fig. 17.7: "Row 0 (1
and 7) ... Row 1 (5, 3, and 9) ... Row 2 (2 and 8) ... Row 3 (6)") and CSC
(Fig. 17.16: "Column 0 (1 and 5) ... Column 1 (7 and 2) ... Column 2 (3 and
8) ... Column 3 (9 and 6)"), which cross-reference into a unique matrix:

```
[1 7 0 0]
[5 0 3 9]
[0 2 8 0]
[0 0 0 6]
```

This also matches §17.4's ELL example (row lengths [2,3,2,1], row 1 is
longest) and §17.3's stated `rowPtrs = [0, 2, 5, 7, 8]`. Files 01, 02, 03,
and 06 use this exact matrix as one of their test cases (with assertions on
the derived `rowPtrs`/`colPtrs`/row-length arrays where the text states
them), in addition to larger randomly generated sparse matrices for
meaningful timing.

**A note on scope.** §17.5's and §17.6's own worked examples (Fig. 17.13's
hybrid matrix, Fig. 17.14's JDS matrix) are described structurally but
without printed numeric values, so files 04 and 05 build their own small
test matrices instead -- as the task brief calls for -- specifically
constructed to exercise each format's key behavior (a few outlier-dense rows
for the hybrid format; widely varying row lengths for JDS). §17.6 states
that "the code for implementing SpMV/JDS is left as an exercise," but (like
§14.3/§14.6/§14.8's identically phrased passages in this project's Chapter
14 samples) the section first walks through the JDS construction and its
coalesced-access physical view (Fig. 17.14, Fig. 17.15) in complete prose
detail; file 05 implements exactly that described kernel. No content from
the numbered "§17.9 Exercises" list itself is implemented anywhere in this
chapter's samples, and §17.7's CSC file implements the SpMV/CSC kernel the
section spells out directly (Fig. 17.18) rather than the transpose/SpMSpV
variants only mentioned in passing at the end of §17.7.

## §17.2 SpMV/COO -- `01_spmv_coo.cu`

The Coordinate (COO) format stores every non-zero as an independent
`(rowIdx[i], colIdx[i], value[i])` triple -- no ordering requirement, so
non-zeros can be appended freely (§17.2's flexibility argument). One thread
per non-zero: each thread looks up its row/column/value, multiplies by
`x[colIdx[i]]`, and **atomically** accumulates into `y[rowIdx[i]]`, since
multiple threads (multiple non-zeros of the same row) can update the same
output element. Memory accesses to the three COO arrays are coalesced
(consecutive threads read consecutive array positions), but the atomics are
COO's main drawback per §17.2.

## §17.3 SpMV/CSR -- `02_spmv_csr.cu`

Compressed Sparse Row (CSR) groups non-zeros by row and replaces COO's
`rowIdx` array with a `rowPtrs` array (size `numRows+1`) giving each row's
start offset. One thread per row: each thread walks its own row's non-zeros
(`rowPtrs[row]` to `rowPtrs[row+1]-1`), accumulates a private dot-product
`sum`, and writes it to `y[row]` once -- **no atomics needed**, since each
row belongs to exactly one thread. The tradeoff (§17.3): consecutive threads
read far-apart `value`/`colIdx` locations each iteration (not coalesced),
and rows of very different lengths cause control divergence within a warp.

## §17.4 SpMV/ELL -- `03_spmv_ell.cu`

ELL starts from CSR's row grouping, pads every row with zero elements up to
the length of the longest row, then lays the now-rectangular matrix out in
**column-major** order (`i = t*numRows + row` for row `row`'s `t`-th
element) -- equivalent to transposing the padded matrix. One thread per row,
same structure as CSR, but now consecutive threads read consecutive
addresses on every loop iteration, so SpMV/ELL's matrix accesses **are**
coalesced. The `nnzPerRow` array lets each thread stop at its row's actual
non-zero count instead of looping over padding. The cost: rows with an
exceptionally large non-zero count force excessive padding on every other
row (§17.4's own example: one outlier row can make ELL storage 20x larger
than CSR).

## §17.5 Hybrid ELL-COO -- `04_spmv_hybrid_ell_coo.cu`

Regulates ELL's padding by capping every row's ELL part at `ellCap`
non-zeros; whatever doesn't fit spills into a separate COO array. SpMV/ELL
(§17.4, capped) runs first and assigns `y[row]`; SpMV/COO (§17.2) then runs
over just the overflow non-zeros and atomically adds its contribution on
top, together computing the full dense matvec. This file builds a test
matrix with a few deliberately much-denser "outlier" rows (mirroring Fig.
17.13's rows 1 and 6) so the regulation has a real effect, and reports the
padded-element count of the capped hybrid representation against what an
uncapped ELL representation would have needed -- e.g. the 400x300 case
needs 3,023 stored elements (ELL+COO combined) versus 79,600 for uncapped
ELL padded to the single densest row's length.

## §17.6 SpMV/JDS -- `05_spmv_jds.cu`

Jagged Diagonal Storage sorts rows from longest to shortest (keeping a
`rows`/`origRow` array to permute the final answer back), then stores the
sorted rows' non-zeros column-major **without padding**: since row length
only decreases going down the sorted order, iteration `t`'s "column" is
exactly the shrinking prefix of rows whose length is `> t`. An `iterPtr`
array (size `maxRowLen+1`) records where each iteration's data begins in the
flattened arrays; a sorted row at position `r` with length `L` has its `t`-th
non-zero at flat index `iterPtr[t] + r`. One thread per sorted row walks its
own non-zeros via this indexing and writes to a `ySorted` array, which the
host then un-permutes into the original row order using `origRow`. Test
matrices use row densities that cycle across a wide range so the sort
actually reorders rows, exercising the reduced-control-divergence benefit
§17.6 describes (adjacent threads land on similar-length rows after
sorting).

## §17.7 SpMV/CSC -- `06_spmv_csc.cu`

Compressed Sparse Column mirrors CSR with rows and columns swapped:
`value`/`rowIdx` grouped by column, `colPtrs` giving each column's start
offset. §17.7 states plainly that "CSC is not intended to be used for
performing SpMV" but works through an SpMV/CSC kernel anyway "for
completeness" (Fig. 17.17/17.18) -- implemented here exactly as described:
one thread per column, loading `x[col]` once (this is CSC's one genuine
SpMV advantage -- a coalesced, single-use read of the input vector), then
walking the column's non-zeros and **atomically** accumulating
`value[i] * x[col]` into `y[rowIdx[i]]`, since different columns (different
threads) can share a row. Per §17.7, this combines COO's atomic-output
drawback with CSR's uncoalesced-matrix-access and control-divergence
drawbacks -- the worst of both, which is exactly why the text calls CSC
unsuitable for SpMV in practice and useful instead for genuinely
column-oriented computations (vector-matrix multiplication, sparse-vector
SpMV) that this file does not implement.

## Results

```
== bin/01_spmv_coo ==
SpMV with COO format, one thread per non-zero, atomic accumulate (§17.2, Fig. 17.5):
canonical 4x4 (nnz=8): 0.0133 ms  [match]
random 500x400 density=0.05 (nnz=10005): 0.0082 ms  [match]
random 1024x1024 density=0.01 (nnz=10441): 0.0073 ms  [match]
PASS
== bin/02_spmv_csr ==
SpMV with CSR format, one thread per row, no atomics (§17.3, Fig. 17.9):
canonical 4x4 (nnz=8, rowPtrs as expected): 0.0092 ms  [match]
random 500x400 density=0.05 (nnz=10005): 0.0338 ms  [match]
random 1024x1024 density=0.01 (nnz=10441): 0.0215 ms  [match]
PASS
== bin/03_spmv_ell ==
SpMV with ELL format, one thread per row, column-major padded storage (§17.4, Fig. 17.12):
canonical 4x4 (maxNnzPerRow=3, as expected): 0.0082 ms  [match]
random 500x400 density=0.05 (maxNnzPerRow=40): 0.0328 ms  [match]
random 1024x1024 density=0.01 (maxNnzPerRow=24): 0.0213 ms  [match]
PASS
== bin/04_spmv_hybrid_ell_coo ==
SpMV with hybrid ELL-COO format: capped ELL + COO overflow for outlier rows (§17.5):
400x300 ellCap=4 trueMaxRow=199 cooOverflow=1423 (padded elems 3023 vs uncapped ELL 79600): 0.0123 ms  [match]
64x64 ellCap=3 trueMaxRow=57 cooOverflow=153 (padded elems 345 vs uncapped ELL 3648): 0.0095 ms  [match]
PASS
== bin/05_spmv_jds ==
SpMV with JDS format: rows sorted by length, unpadded column-major storage (§17.6):
300x250 maxRowLen=146 nnz=20181 (rows sorted descending): 0.1259 ms  [match]
1024x512 maxRowLen=303 nnz=140966 (rows sorted descending): 0.2505 ms  [match]
PASS
== bin/06_spmv_csc ==
SpMV with CSC format, one thread per column, atomic accumulate (§17.7, Fig. 17.18):
canonical 4x4 (nnz=8, colPtrs as expected): 0.0113 ms  [match]
random 500x400 density=0.05 (nnz=10005): 0.0563 ms  [match]
random 1024x1024 density=0.01 (nnz=10441): 0.0325 ms  [match]
PASS
```

All six binaries also pass under `compute-sanitizer --tool memcheck` (0
errors) at `-arch=sm_75` -- notable here since files 01, 04, and 06 rely on
`atomicAdd` into shared output locations.

Build and run all samples in this chapter:

```sh
make run
```
