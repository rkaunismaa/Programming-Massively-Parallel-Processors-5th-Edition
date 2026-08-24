# Chapter 3: Multidimensional grids and data

Source: *Programming Massively Parallel Processors*, 5th ed., Ch. 3 (pp. 45-64).

| File | Book section | Technique |
|------|-------------|-----------|
| `01_rgb_to_grayscale.cu` | §3.2 | `colorToGrayscaleConversion` (Fig. 3.4): 2D grid of 2D blocks, one thread per pixel, `row`/`col` from `blockIdx`/`blockDim`/`threadIdx`, RGB → grayscale via `L = 0.299r + 0.587g + 0.114b` |
| `02_image_blur.cu` | §3.3 | `blurKernel` (Fig. 3.8): one thread per output pixel, averages a `(2*BLUR_SIZE+1)x(2*BLUR_SIZE+1)` neighborhood (3x3, `BLUR_SIZE=1`), with the `curRow`/`curCol` in-bounds guard that clips patches at the image edges (Fig. 3.9) |
| `03_matrix_multiplication_naive.cu` | §3.4 | `matrixMulKernel` (Fig. 3.11): 2D grid, one thread per output element, inner product of a row of `M` and a column of `N` |

§3.1 (Multidimensional grid organization) is conceptual: it explains that a
grid is a 3D array of blocks and each block a 3D array of threads, the `dim3`
execution configuration parameters, and how `blockIdx`/`threadIdx`/`gridDim`/
`blockDim` give a thread its coordinates (Fig. 3.1's toy `(2,2,1)` grid of
`(4,2,2)` blocks). It has no standalone kernel of its own, so it is folded
into the background here rather than given a separate file. All three
samples below build directly on it: each maps a 2D grid of 16x16 blocks onto
2D image or matrix data using
```
row = blockIdx.y*blockDim.y + threadIdx.y
col = blockIdx.x*blockDim.x + threadIdx.x
```
and each guards its work with an `if (row < height && col < width)` (or
`Width`) test, exactly as introduced for `colorToGrayscaleConversion` in
§3.2 and reused unchanged in `blurKernel` (§3.3) and `matrixMulKernel`
(§3.4). §3.2 also covers linearizing (flattening) dynamically allocated 2D
arrays into row-major 1D storage (Fig. 3.3); all three samples index their
buffers this way.

Each sample generates its own synthetic input (an RGB or grayscale buffer,
or a pair of matrices), computes a CPU reference with the same formula and
loop order as the kernel, runs the CUDA kernel, and compares the two. Image
sizes (`403x251`) and the matrix size (`500`) are deliberately not multiples
of the 16x16 block size, so every run exercises the partial/boundary blocks
the book discusses (Fig. 3.5 for grayscale, Fig. 3.9 for blur's corner/edge
pixels). The grayscale comparison allows an off-by-one byte, to account for
possible floating-point rounding differences (e.g. FMA contraction) between
host and device evaluation of the same formula; blur's comparison is exact,
since all of its arithmetic is integer; matmul uses `nearlyEqual` from
`common/cuda_utils.h`, appropriate for its accumulated floating-point sums.

Build and run all samples in this chapter:

```sh
make run
```
