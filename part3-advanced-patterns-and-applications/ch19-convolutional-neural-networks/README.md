# Chapter 19: Convolutional neural networks

Source: *Programming Massively Parallel Processors*, 5th ed., Ch. 19 (pp. 453-475).

| File | Book section | Technique |
|------|-------------|-----------|
| `01_cnn_conv_layer_direct.cu` | §19.2 | Direct convolutional-layer kernel: one thread per output pixel, TILE_WIDTH x TILE_WIDTH thread blocks, `gridDim(M, T, N)` (Fig. 19.7) |
| `02_cnn_conv_layer_as_gemm.cu` | §19.3 | Convolutional layer as GEMM: tiled matrix multiplication with the input feature map matrix `B` unfolded implicitly, tile-by-tile, straight from `X` (Fig. 19.11) |

Both files implement the forward (inference) path of a batched convolutional
layer: input feature maps `X` are an `N*C*H*W` tensor, filters `F` are an
`M*C*K*K` tensor, and output feature maps `Y` are an `N*M*H_out*W_out`
tensor with `H_out = H-K+1`, `W_out = W-K+1` -- a "valid" convolution with no
padding (§19.1: LeNet-5 simply treats the missing right/bottom border pixels
as ghost cells rather than assuming any padding convention), and no
activation function (§19.1 explicitly narrows the chapter's scope to
generating the convolution result itself). Each file builds its own random
test input, computes its own CPU reference (the straightforward nested-loop
`cpuConvLayerForward` from Fig. 19.4), runs its CUDA kernel(s), and compares
with `nearlyEqual`, printing `PASS`/`FAIL` plus a timing line per this
repo's convention -- no code is shared between the two files.

**Book-exact worked example.** Both files additionally hardcode the book's
own tiny example (§19.1, Fig. 19.2b/Eq. 19.1 and §19.3, Fig. 19.8/Eq.
19.2-19.3): `C=3` 3x3 input feature maps, `M=2` output feature maps, `K=2`
filters, whose corner output pixel is worked out by hand in the text to be
`14`. File 01 checks its CPU reference against the full expected output
(`[[14,20],[15,24]]` and `[[12,24],[17,26]]`); file 02 additionally checks
that explicitly unfolding `X` into the conceptual `B` matrix (Eq. 19.5) and
multiplying `F * B` on the CPU reproduces the identical values, i.e. that
the GEMM reformulation itself is correct independent of the tiled GPU
kernel. This example is too small to exercise either GPU kernel directly,
since both assume (per the book, which explicitly omits bounds checking and
"leaves it as an exercise") that the relevant dimensions divide evenly by
`TILE_WIDTH=16`; it is used purely as a CPU-side cross-check, and the actual
GPU-vs-CPU `PASS`/`FAIL` verdict comes from larger, `TILE_WIDTH`-friendly
generated test cases (one of which, in file 01, reproduces Fig. 19.6's own
grid-mapping example exactly: `M=4` output feature maps, each tiled as a 2x2
grid of `TILE_WIDTH=16` tiles).

**A note on scope.** §19.4 (the CUDNN library) is a discussion of a
third-party library's API and algorithm choices, not a from-scratch kernel
the chapter presents -- there is nothing in it to implement -- and no system
CUDNN development package is installed in this environment, so no cuDNN
sample is included here.

## §19.1, §19.5-19.6 Notes not given a separate file

- **§19.1 Convolutional neural networks**: introduces CNNs via LeNet-5
  (Fig. 19.1, handwritten-digit recognition) and gives the sequential
  reference C code both this chapter's files build on -- a single-sample
  `convLayer_forward` (Fig. 19.3) and its batched generalization,
  `convLayer_batched` (Fig. 19.4, one additional outer loop over the `N`
  samples in a mini-batch) -- which is the version both `01` and `02`
  actually implement on the GPU. The section also explicitly scopes the
  whole chapter to the *forward* path only, even though it's discussing a
  *training*-relevant computation: "the forward path is activated during
  both the training phase and the inference phase. We will therefore use
  the forward path to illustrate the acceleration" -- backward propagation
  (training) is deferred to Appendix B, matching why Exercise 3 (implement
  the backward pass) is out of scope here.
- **§19.5 Summary / §19.6 Exercises**: recap, plus the numbered exercises
  -- implementing a subsampling layer's forward pass and the convolutional
  layer's backward pass (both deferred to Appendix B material, outside
  this chapter's own scope), a layout-choice discussion question
  (`[N*C*H*W]` vs. `[N*H*W*C]` vs. `[C*H*W*N]`, left open here as a genuine
  design tradeoff rather than one with a single derivable answer), and the
  Fig. 19.11 memory-access analysis, which *is* answered above rather than
  left as an open exercise.

## §19.2 Direct convolutional-layer kernel -- `01_cnn_conv_layer_direct.cu`

Each thread computes one output pixel `Y[n,m,h,w]`. Thread blocks are
`TILE_WIDTH x TILE_WIDTH` (`TILE_WIDTH=16`), each computing one tile of one
output feature map, and the 3D grid is `(M, T, N)` (Fig. 19.5): `blockIdx.x`
selects the output feature map, `blockIdx.z` selects the sample in the
batch, and `blockIdx.y` linearizes the `H_grid * W_grid` tiles within a
feature map into a single dimension (`T = H_grid*W_grid`), since only one
grid dimension is left once `X` and `Z` are claimed for the feature-map and
batch indices (Fig. 19.6). Each thread then sums a convolution between a
`KxK` patch of every input channel and the matching filter, accumulating
across all `C` input channels into `acc` before writing the result (Fig.
19.7). No shared-memory tiling of `X` or `F` is done, so the kernel is
limited by global memory bandwidth -- exactly the drawback §19.3 sets out to
fix.

## §19.3 Convolutional layer as GEMM -- `02_cnn_conv_layer_as_gemm.cu`

§19.3 recasts the layer as one matrix multiplication `Y = F * B`: `F`, laid
out `M*C*K*K` in memory, is already usable directly as an `M x (C*K*K)`
matrix with no rearrangement, and the conceptual `B` matrix (`(C*K*K) x
(H_out*W_out)`, one column per output pixel) is formed by unfolding and
duplicating overlapping patches of `X`. The chapter shows that materializing
`B` explicitly in global memory is wasteful (Eq. 19.4: an expansion ratio of
`K^2 * H_out*W_out / (H_in*W_in)` over `X`, easily 20x or more for realistic
layer sizes) and motivates an implicit approach instead: a tiled
matrix-multiplication kernel (adapted from the Chapter 5 tiled MM kernel)
that loads each `B` tile on demand straight from `X`, mapping each
conceptual `B[u,v]` element back to its source `X` element via Eq. (19.5),
never materializing `B` in global memory (Fig. 19.9/19.10/19.11). This file
implements `ConvLayer_MM_Kernel` (Fig. 19.11) unchanged, including its
stated assumption that `M` and `H_out*W_out` divide evenly by `TILE_WIDTH`
(no bounds checking, per the book).

**Exercise 4's analysis, worked out.** §19.6 Exercise 4 asks the reader to
analyze Fig. 19.11's memory access pattern for coalescing and bank
conflicts, without a kernel to write -- answered here rather than left
open, and checked numerically (not just asserted) before writing it down:

- **Global memory (the `Bds` load, line 28).** `TILE_WIDTH=16` means
  `blockDim.x=16 < 32`, so one warp spans **two** `ty` values (`ty=0` for
  lanes 0-15, `ty=1` for lanes 16-31 -- CUDA linearizes `threadIdx.y` above
  `threadIdx.x`). Within either half-warp, `ty` (and hence `u =
  ph*TILE_WIDTH+ty`) is fixed and only `v = Col` varies by 1 per lane, so
  Eq. (19.5)'s index increases by exactly 1 per lane -- coalesced --
  *unless* the 16 consecutive `v` values straddle an output-row boundary
  (`v` crossing a multiple of `W_out`), which can split one half-warp's
  access into two separate contiguous runs instead of one (a small
  discontinuity, not a scattered access). The two half-warps themselves
  read from different, usually nearby, `u`-based regions of `X`. Net: good
  coalescing overall, but not the single trivially-provable transaction a
  simple row-major read would give -- exactly the nuance the exercise is
  testing for.
- **Shared memory.** The `Bds[k][tx]` read in the accumulation loop is the
  standard tiled-matmul pattern (fixed `k`, `tx` fast-varying -> consecutive
  banks): conflict-free. The `Fds[ty][k]` read is a broadcast within each
  `ty`-group (16 lanes share one address), and the two groups' addresses
  (`ty*TILE_WIDTH+k` for `ty` in `{0,1}`) are always exactly 16 apart, hence
  always in *different* banks for every `k` (verified by brute force over
  `k=0..15`, not just argued) -- both groups broadcast simultaneously with
  no conflict. **Conclusion: no bank conflicts anywhere in this kernel.**

## Results

```
== bin/01_cnn_conv_layer_direct ==
Direct CUDA convolutional layer kernel, one thread per output pixel (§19.2, Fig. 19.7):
book example (§19.1, Fig. 19.2b): C=3 3x3, K=2, M=2 -> Y[0,0,0,0]=14 [match]
Fig 19.6 grid example (2x2 tiles/map): N=2 C=3 H=36 W=36 K=5 M=4 -> H_out=32 W_out=32: 0.0768 ms  [match]
single-tile config: N=1 C=2 H=18 W=18 K=3 M=6 -> H_out=16 W_out=16: 0.0225 ms  [match]
PASS
== bin/02_cnn_conv_layer_as_gemm ==
Convolutional layer as GEMM: tiled matrix multiplication with implicit B-matrix unfolding (§19.3, Fig. 19.11):
book example (§19.3, Fig. 19.8): B is 12x4 (expansion ratio 1.778x over X, Eq. 19.4) -> Y[0,0,0,0]=14 [match]
multi-phase config: N=2 C=3 H=11 W=11 K=4 M=32 -> H_out=8 W_out=8, 3 phase(s): 0.0450 ms  [match]
single-tile config: N=1 C=1 H=7 W=7 K=4 M=16 -> H_out=4 W_out=4, 1 phase(s): 0.0182 ms  [match]
PASS
```

Both binaries also pass under `compute-sanitizer --tool memcheck` (0
errors) at `-arch=sm_75`.

Build and run all samples in this chapter:

```sh
make run
```
