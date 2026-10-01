# Chapter 3 — Multidimensional Grids and Data

Section-by-section summary of the book content. Page numbers are printed
book pages (Chapter 3 runs pp. 45–66).

## 3.1 Multidimensional grid organization (p. 45)

A grid is, in general, a **three-dimensional array of blocks**, and
each block is a **three-dimensional array of threads**. The two
execution-configuration parameters in a kernel call (`<<<...>>>`) each
have type **`dim3`** — a struct of three unsigned integer fields
`x`/`y`/`z` — specifying the grid's dimensions (in blocks) and each
block's dimensions (in threads), respectively; unused dimensions are
set to 1. `dimGrid`/`dimBlock` are ordinary host-code variables chosen
by the programmer (any name, as long as the type is `dim3`) and can be
computed from other variables (e.g. `dim3 dimGrid(ceil(n/256.0), 1,
1)`), letting grid size scale with data size. CUDA also offers a
1D-only shorthand: passing plain integers (as in Fig. 2.12's
`vecAddKernel<<<ceil(n/256.0), 256>>>(...)`) relies on `dim3`'s
constructor defaulting unspecified fields to 1, so a bare integer
becomes the `x` dimension with `y`/`z` implicitly 1.

Inside the kernel, the built-in `gridDim` and `blockDim` variables are
pre-initialized from these execution-configuration parameters (fixed
names, part of the CUDA C++ specification — unlike the host-side
`dimGrid`/`dimBlock` variable names, which are the programmer's
choice). Allowed ranges: `gridDim.x` from 1 to 2³¹−1; `gridDim.y` and
`gridDim.z` from 1 to 2¹⁶−1 (65,535). A block's total thread count
(`blockDim.x * blockDim.y * blockDim.z`) is capped at 1024 regardless
of how it's distributed across the three dimensions — e.g. (512,1,1),
(8,16,4), and (32,16,2) are all legal, but (32,32,2) (=2048) is not. A
grid and its blocks don't need matching dimensionality — a grid can be
organized in fewer or more dimensions than its own blocks (Fig. 3.1).

Fig. 3.1's small example has `gridDim=(2,2,1)` and `blockDim=(4,2,2)`.
Blocks are conventionally labeled `(blockIdx.y, blockIdx.x)` —
**highest dimension first**, the reverse of the `dim3(x,y,z)`
constructor order — because that reversed convention turns out to read
more naturally once thread coordinates are mapped onto multidimensional
*data* indices (§3.2), where the book also orders dimensions
highest-first (e.g. an `n×m` picture means `n` rows, `m` columns —
matching ordinary row/column notation).

## 3.2 Mapping threads to multidimensional data (p. 48)

Thread organization (1D/2D/3D) is chosen to match the data's own
natural dimensionality — a 2D picture is naturally processed with a 2D
grid of 2D blocks. Worked example: a 62×76 picture `Pin` (62 rows, 76
columns) processed with 16×16 blocks (Fig. 3.2) needs `ceil(62/16)=4`
blocks vertically and `ceil(76/16)=5` horizontally — 20 blocks, 64×80
threads total, 2 extra rows and 4 extra columns of threads beyond the
image's actual size (mirroring the 1D vector-addition boundary case
from Chapter 2 — the kernel needs an `if` guard to disable those extra
threads).

Each thread derives its own row/column from its indices:
`row = blockIdx.y*blockDim.y + threadIdx.y`,
`col = blockIdx.x*blockDim.x + threadIdx.x`. The host launches this
with `dim3 dimGrid(ceil(m/16.0), ceil(n/16.0), 1); dim3
dimBlock(16, 16, 1);` for an `n`-row, `m`-column picture — note `m`
(columns, the x/horizontal extent) drives the grid's `x` dimension and
`n` (rows, y/vertical) drives its `y` dimension, the opposite order
from how the book names the picture itself (`n×m`, rows first).

**Linearization:** C/C++ ultimately store all arrays — including
multidimensional ones — in a single "flat" memory space (sidebar), so
a 2D element access like `Pin_d[j][i]` only compiles when the number
of columns is known at *compile* time; a dynamically-sized 2D array
(size decided at run time) must be manually linearized into an
equivalent 1D index. The book uses **row-major layout** (Fig. 3.3):
all elements of row 0, then all of row 1, etc. — giving element
`M_{j,i}` (row `j`, column `i`, `Width` columns per row) the 1D index
`j*Width + i` (e.g. `M_{2,1}` in a 4-wide matrix → `2*4+1 = 9`). This
is the layout C/C++ compilers use for statically-sized arrays
automatically, and the layout CUDA C++ programmers must replicate by
hand for dynamically-allocated ones. (Column-major — used by Fortran —
is mentioned only to note it's the transpose of row-major, and that
libraries built for Fortran callers often expect transposed input.)

Applying this to `colorToGrayscaleConversion` (Fig. 3.4): the
grayscale output is one byte per pixel, so its linear offset is simply
`grayOffset = row*width + col`; the RGB input has 3 consecutive bytes
per pixel, so its linear offset is `rgbOffset = grayOffset*CHANNELS`
(`CHANNELS = 3`), from which `r`/`g`/`b` are read at
`rgbOffset`/`rgbOffset+1`/`rgbOffset+2` before applying the same
`L = 0.299*r + 0.587*g + 0.114*b` weighting as Chapter 2 (printed as
`0.21f*r + 0.71f*g + 0.07f*b` in the code, with minor rounding of the
same weights). Fig. 3.5 works through all 20 blocks of the 62×76
example by hand, sorting them into 4 cases based on which of
`row`/`col`'s bounds checks fail for which threads (fully in-range
blocks process all 256 threads; blocks straddling the right edge waste
4 threads/row; blocks straddling the bottom edge waste 2 threads/column;
the single corner block wastes both) — illustrating concretely that
boundary-check "waste" is proportional to image edges, not image area,
and so shrinks relative to total work as images get larger.

Extending to **3D** data just adds one more level of linearization:
`plane = blockIdx.z*blockDim.z + threadIdx.z`, with a 3D array `P`
accessed at the fully-linearized `P[plane*m*n + row*m + col]`, needing
all three (`plane`, `row`, `col`) bounds-checked. (3D arrays are used
more fully for the stencil pattern in Chapter 8.)

## 3.3 Image blur — a more complex kernel (p. 55)

Introduces a kernel where each thread does more than one simple
arithmetic operation on one element — a step up from vector
addition/grayscale conversion toward the book's later, more
sophisticated patterns. **Image blurring** (Fig. 3.6) smooths out
abrupt pixel-value variation — useful for reducing visual noise,
helping edge-detection/object-recognition algorithms ignore
fine-grained clutter, or deliberately drawing visual focus by blurring
everything *except* one region. Mathematically, it computes each output
pixel as a (here, unweighted) average over a surrounding `N×N` patch of
input pixels, centered on and including the target pixel — a
simplified instance of the general **convolution** pattern formalized
in Chapter 7 (where, unlike here, patch elements are weighted
differently based on distance from the center, e.g. Gaussian blur).

Fig. 3.8's `blurKernel` keeps the same `row`/`col` thread-to-pixel
mapping as the grayscale kernel, but instead of one memory read, each
thread runs a **nested loop** (`blurRow`, `blurCol`, each ranging
`-BLUR_SIZE` to `+BLUR_SIZE`) over its own patch, accumulating a
running sum `pixVal` and a running count `pixels` of how many patch
pixels were actually valid, then dividing to get the average
(`BLUR_SIZE` = the patch's radius on each side; e.g. `BLUR_SIZE=1` for
a 3×3 patch, `BLUR_SIZE=3` for 7×7). The inner `if (curRow>=0 &&
curRow<h && curCol>=0 && curCol<w)` check (Fig. 3.7/3.9) is what makes
this "more complex": unlike the grayscale kernel's single boundary
check on the *output* pixel, here *every one of the up-to-nine input
reads per thread* needs its own independent bounds check, since a
patch centered near an image edge or corner legitimately has some
neighbors that don't exist. Fig. 3.9 works through the corner/edge
cases by hand: a corner pixel's patch has only 4 valid pixels (of 9),
an edge pixel's has 6, and only interior pixels get the full 9 — this
is exactly why the running `pixels` counter (rather than a hard-coded
divisor) is needed, so the average is computed correctly regardless of
how many of the patch's pixels actually existed.

## 3.4 Matrix multiplication (p. 59)

**Matrix multiplication** is introduced as a BLAS (Basic Linear Algebra
Subprograms) level-3 operation (sidebar: level-1 = vector-vector,
e.g. Chapter 2's vector addition is a degenerate level-1 BLAS case with
`α=1`; level-2 = matrix-vector, covered in Chapter 17's sparse linear
algebra; level-3 = matrix-matrix, `C = αAB + βC`, of which plain matrix
multiplication is the `α=1, β=0` special case) — important both as a
building block for higher-level linear algebra (e.g. LU decomposition)
and directly for deep learning (Chapters 19–20).

Multiplying an `i×j` matrix `M` by a `j×k` matrix `N` produces an
`i×k` matrix `P`, where each output element `P_{row,col}` is the
**dot product** of `M`'s `row`-th row and `N`'s `col`-th column:
`P_{row,col} = Σ M_{row,k} * N_{k,col}` for `k = 0..Width-1` (Fig. 3.10).
The CUDA mapping mirrors the grayscale kernel exactly: one thread per
output element, same `row`/`col` index calculations, same bounds
check — but this chapter's kernel (Fig. 3.11) simplifies to **square**
matrices (a single `Width` parameter rather than separate `i`/`j`/`k`
dimensions), with the dot product itself computed by an inner
`for (k=0; k<Width; ++k) Pvalue += M[row*Width+k]*N[k*Width+col];`
loop — `M`'s row-th row is contiguous in its row-major linearization
(`M[row*Width+k]`, stepping by 1 per `k`), while `N`'s col-th column is
*not* contiguous (`N[k*Width+col]`, stepping by a full `Width` per
`k`), since accessing a column means skipping over whole rows (this
asymmetry becomes important for memory-coalescing reasoning in
Chapter 6).

This thread-to-data mapping effectively divides the output `P` into
tiles, one tile per block (Fig. 3.10's large light-colored square).
Fig. 3.12's small 4×4 example with `BLOCK_WIDTH=2` is traced fully by
hand: Thread (0,0) of Block (0,0) computes `P_{0,0}`; Thread (0,0) of
Block (1,0) computes `P_{2,0}`; and Fig. 3.13 walks Thread (0,0) of
Block (0,0) through all four loop iterations explicitly, confirming
each `k` accesses the correct linearized `M`/`N` elements and that the
final `Pvalue` written to `P[0]` is indeed the dot product of `M`'s row
0 and `N`'s column 0.

Grid/block size limits (§3.1) cap how large an output matrix a single
kernel launch of this form can cover; larger problems need either
multiple grid launches over sub-matrices, or a kernel redesigned so
each thread computes multiple `P` elements — both directions explored
further in Chapters 5 and 15.

## 3.5 Summary (p. 64)

CUDA grids and blocks can each be organized in up to three dimensions,
useful for mapping naturally multidimensional data onto threads via
each thread's unique `blockIdx`/`threadIdx` coordinates. Because
dynamically-allocated multidimensional arrays are, under the hood,
always linearized (row-major) one-dimensional arrays in C/C++, kernel
code must compute that linear offset itself from a thread's
multidimensional coordinates. The chapter worked through three
examples of increasing complexity — grayscale conversion, image blur,
matrix multiplication — to build the mechanical skill of reasoning
about multidimensional grids processing multidimensional data, which
the rest of the book's parallel patterns and optimizations build on.

Four exercises (not numbered/listed separately here since they overlap
with the "Exercises" count below) ask the reader to: (1) design,
implement, and compare row-per-thread and column-per-thread matrix
multiplication kernel variants against the chapter's
element-per-thread version; (2) implement a matrix-vector
multiplication kernel; (3) read off thread/block counts and
participating-thread counts from a given 2D kernel and its launch
configuration; (4)/(5) compute linearized 1D array indices by hand for
given 2D (row-major and column-major) and 3D (row-major) examples. Not
implemented in this repo's samples (end-of-chapter exercises are out of
scope — see the repo root README).
