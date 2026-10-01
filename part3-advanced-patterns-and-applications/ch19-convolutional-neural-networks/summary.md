# Chapter 19 — Convolutional Neural Networks

Section-by-section summary of the book content. Page numbers are printed
book pages (Chapter 19 runs pp. 453–475).

## 19.1 Convolutional neural networks (p. 453)

**CNNs**, invented in the late 1980s, are feed-forward networks first
applied experimentally (early 1990s) to OCR, handwriting, speech, and
face recognition — but largely sidelined through the 1990s–2000s by
hand-engineered computer-vision features, insufficient labeled data,
and a prevailing belief that learning deep hierarchical feature
extractors from data was computationally infeasible. Deep, feed-forward
networks were revived around 2006 via unsupervised pretraining methods,
with the first major success in **speech recognition** — made
practical by GPUs training networks ~10× faster than CPUs. CNNs
specifically broke through in **2012**, when a University of Toronto
team's network (60M parameters, 650,000 neurons, trained on 1.2M
ImageNet images across two GPUs in one week, using Alex Krizhevsky's
CUDA-based CNN library) won the ILSVRC contest with a 15.3% error rate
versus the runner-up's 26.2% (traditional computer-vision methods) —
triggering the broader deep-learning revolution.

The chapter's running example is **LeNet-5** (Fig. 19.1), a late-1980s
CNN for handwritten-digit recognition, composed of three layer types:
**convolutional**, **subsampling**, and **fully connected** — the
convolutional layers dominate execution time and are the chapter's
focus (other layer types are deferred to Appendix B). In LeNet-5's
**forward path**, input flows left-to-right from a 32×32 grayscale
image to a 10-element output giving the probability of each of 10
digit classes. The forward path is used both for **inference**
(interpreting the output as the final answer) and, during
**training**, to produce the output compared against the true label —
with disagreement triggering the **backward-propagation path** to
adjust parameters (covered in Appendix B; this chapter's acceleration
work focuses entirely on the shared forward path).

## 19.2 A CUDA convolutional layer kernel (p. 456)

Layer inputs/outputs are called **feature maps** (or **features**).
Each convolutional layer output pixel is produced by convolving a
small local patch of **input** feature maps with learned weights (a
**filter**, Chapter 7's sense) — conceptually, one can think of a
convolutional layer as a collection of **perceptrons**, each taking a
patch of input pixels and an activation function (sigmoid, etc.); the
chapter focuses purely on efficiently computing the **convolution
result** itself, treating activation functions as trivially fusable
afterward. Since a layer generally has multiple input *and* multiple
output feature maps, a separate filter exists **per (input, output)
feature-map pair** — Fig. 19.2 works a small 3-input/2-output example
(6 filters total — C×M); LeNet's C3 layer similarly uses 6×16=96
filters. Each output pixel is the **sum of convolutions across all
input feature maps** (Fig. 19.2b).

Fig. 19.3's `convLayer_forward` is a direct sequential C implementation:
input `X` is a 3D `C×H×W` array (channel, height, width); output `Y` is
3D `M×H_out×W_out` (`H_out = H-K+1`, `W_out = W-K+1` — note no explicit
padding is used, so output shrinks per layer, matching LeNet-5's
convention of treating right/bottom edge pixels as implicit "ghost
cells," shrinking each dimension by 4 for a 5×5 filter); filters `F` are
a 4D `M×C×K×K` array. Five nested loops: output feature map (`m`),
output row/column (`h`,`w`), then **serial** (not parallelized here)
input-channel (`c`) and filter-position (`p`,`q`) loops accumulating the
convolution sum — the book notes the innermost three loops *could* be
parallelized too, but doing so would need atomic accumulation into
shared `Y` elements across iterations, so the chapter keeps them
serial unless genuinely needed.

**Batched execution**: CNN training/inference processes many input
samples together as a **batch**, both to better utilize GPU execution
resources (far more thread blocks available to launch) and because
training naturally divides a dataset into batches. Fig. 19.4's
`convLayer_batched` adds an outer loop over batch samples `n`, with `X`
and `Y` gaining a batch dimension (`F`, the shared filter bank, does
not — identical weights apply across all samples in a batch).

With four "easy" parallel loop levels (`n`, `m`, `h`, `w` — their total
iteration count is `N·M·H_out·W_out`), this maps directly onto a CUDA
grid: one thread per output pixel, 2D thread blocks of
`TILE_WIDTH×TILE_WIDTH` pixels (e.g. 16×16=256 threads) capturing the
`h`-`w` parallelism within one output tile. Grid organization (one of
several valid choices, presented in detail — Figs. 19.5–19.6): grid X
dimension = output feature map index `M` (`blockIdx.x`); grid Z
dimension = batch sample index `N` (`blockIdx.z`); grid Y dimension
**linearizes** the 2D tile position within an output feature map
(`blockIdx.y` decomposed via `/W_grid` and `%W_grid` into vertical/
horizontal tile indices), since CUDA grids only have 3 dimensions but
this scheme needs 4 logical axes (n, m, tile-row, tile-col). Fig. 19.7's
resulting kernel: each thread computes its `(n,m,h,w)` indices from
block/thread indices, then runs the same serial `c`/`p`/`q` triple loop
as the sequential version, accumulating into a private register `acc`
before one final write to `Y`.

This kernel has abundant parallelism but — like the basic convolution
kernel in Chapter 7 — is **global-memory-bandwidth-limited**; shared-
memory tiling (as in Chapter 7) would help but is left as an exercise,
since §19.3 instead presents a more systematic bandwidth-reduction
approach.

## 19.3 Formulating convolutional layer as GEMM (p. 463)

A convolutional layer offers rich reuse opportunities (e.g. each input
feature map is reused across all output feature maps), but the
natural per-output-feature-map thread-block organization (§19.2)
prevents a straightforward shared-memory tiling reuse scheme across
*different* blocks. Instead (following Chellapilla et al.), the chapter
reformulates the entire convolutional layer as one **General Matrix
Multiply (GEMM)** — since matrix multiplication is a dot product
between row vectors of one input and column vectors of another, and a
convolutional layer's output pixel is *also* a dot product (between a
filter bank's weights and an input patch), the filters can be arranged
as one matrix's rows and the (duplicated, "unfolded") input patches as
the other matrix's columns, so GEMM's output *is* the convolution
result — enabling direct use of highly-optimized GEMM kernels (e.g.
cuBLAS) and Chapter 15's advanced tiling techniques.

Concretely (worked small example, Fig. 19.8, C=3, M=2, 2×2 filters):
the **filter-bank matrix F** is formed by linearizing each filter
(row-major) as one matrix row, with filters ordered output-feature-map-
major then input-feature-map-minor — this ordering happens to **match**
the filter array's natural row-major storage order exactly, so **F
needs no rearrangement at all**. The **unfolded input-feature matrix B**
is formed by linearizing (row-major) each output pixel's required input
patch — one per input feature map — into one column, concatenated
across input feature maps into a single column vector; since each
output feature map needs `H_out×W_out` pixels, B has that many columns.
Multiplying `F × B` yields the output feature matrix `Y`, in exactly
the row-major layout later layers expect.

**Cost of explicitly unfolding** (and the implicit alternative): B's
explicit construction **duplicates** input pixels, since overlapping
convolution patches share pixels — the book derives the general
expansion-ratio formula `K²·H_out·W_out / (H_in·W_in)` (Eq. 19.4),
showing the ratio approaches `K²` for large feature maps relative to
filter size — easily 20× or more in practice, which would itself cost
significant extra memory bandwidth, defeating the whole purpose.
Practical implementations instead generate B's tiles **implicitly, on
demand**, exactly as needed by a **tiled matrix multiplication** kernel
(Chapter 5's Fig. 5.9 style) — B is never materialized in global
memory at all, just loaded tile-by-tile straight from the original `X`
array. Figs. 19.9–19.10 derive the general index mapping from a
conceptual `B[u,v]` element to the actual `X` array location it should
be loaded from (Eq. 19.5): decomposing `v` (output pixel's row-major
index) into `(v/W_out, v%W_out)` locates the output pixel (hence the
patch's starting position), and decomposing `u` (patch element's
row-major index within its concatenated multi-channel patch) via
`u/(K²)` (channel), `(u%K²)/K` (row within patch), `(u%K²)%K` (column
within patch) — combined, these give the exact `X[n, channel, row,
col]` element to load for any `B[u,v]`.

Fig. 19.11 presents the complete CUDA kernel: adapted directly from
Chapter 5's tiled matmul kernel (Fig. 5.9), loading `F` tiles exactly as
before, but loading `Bds` (the B tile) via the Eq. 19.5 index-mapping
formula applied to `X` directly (no bounds checking included, left as
an exercise, since the focus is the B-mapping adaptation itself). After
all phases, each thread writes its accumulated `Pvalue` to `Y` using
the batch, row, and column indices.

Matrix multiplication formulated this way is efficient on GPUs
specifically because its data-reuse-per-byte scales with the matrices'
*own* dimensions (Chapter 5) — and while individual convolution
parameters (`C`, `K`, etc.) can be small, the **products** that
determine F's and B's actual matrix dimensions (`M×(C·K·K)` and
`(C·K·K)×(H_out·W_out)`) tend to stay large throughout a network (early
layers: small `C`, large `H_out·W_out`; late layers: large `C`, small
`H_out·W_out` — the product `C·H_out·W_out` stays consistently large
either way), keeping GPU utilization and execution speed high for
every layer.

## 19.4 CUDNN library (p. 472)

**cuDNN** is NVIDIA's library of optimized deep-learning primitives,
providing a flexible, thread-safe C-language API that integrates into
existing frameworks (PyTorch, Caffe, TensorFlow, MXNet, etc.), requiring
GPU-resident input/output data (as cuBLAS does). Tensors/filters are
accessed through opaque descriptors supporting arbitrary per-dimension
strides. cuDNN's most important convolution primitive is a general
**batched convolution**, parameterized (Table 19.1) by batch size `N`,
input/output channel counts `C`/`K`, input height/width `H`/`W`, filter
height/width `R`/`S`, vertical/horizontal **stride** `u`/`v` (letting
the user compute only a subset of output pixels, reducing compute
load), and zero-**padding** amounts `pad_h`/`pad_w` (for memory-
alignment/vectorization benefits) — output dimensions `P`/`Q` are
derived from all of these.

cuDNN supports multiple convolution algorithms (matrix-multiplication-
based/GEMM, Winograd, FFT-based). Its GEMM-based algorithm closely
follows §19.3's approach, but goes further to avoid §19.3's
materialization cost entirely: it **lazily generates and loads the
expanded input-feature matrix directly into on-chip memory**, inside a
general, highly-tuned tiled matrix-multiplication routine (fixed-size
sub-tiles of both operand matrices are read into on-chip caches and
used to compute output sub-tiles, with the next tiles' loads overlapped
behind ongoing arithmetic to hide memory latency, so overall speed is
limited only by compute time). Since the convolution-to-matrix tile
mapping is independent of the matmul routine's own tiling scheme, cuDNN
computes this mapping **dynamically** as the computation proceeds —
costing some extra indexing arithmetic, but fully leveraging the
underlying highly-optimized matmul engine. After computing, cuDNN
performs any needed **tensor transposition** to deliver results in the
caller's requested data layout.

## 19.5 Summary (p. 474)

The chapter built directly on Chapter 7's convolution pattern to present
a basic CUDA convolutional-layer kernel, then showed how formulating
convolutional layers as **matrix multiplication** (via *implicit*
unfolding of the input feature map, avoiding the cost of explicit
materialization) lets convolutional layers benefit from highly
optimized GEMM libraries — including presenting a concrete matrix-
multiplication kernel performing this implicit unfolding directly.
Finally, **cuDNN** was introduced as the production library (used by
virtually every deep-learning framework) that implements these and
further optimizations, letting framework users benefit from
highly-tuned layer implementations without writing their own CUDA
kernels.

## 19.6 Exercises (p. 474)

Four problems: implement the forward pass for LeNet-5's **subsampling**
layer (introduced in §19.1, detailed in Appendix B) (Q1); analyze
whether switching the chapter's `[N×C×H×W]` input/output layout to
`[N×H×W×C]` (or `[C×H×W×N]`) could reduce memory bandwidth, and discuss
potential benefits of each (Q2); implement the **backward**
(back-propagation) pass for the convolutional layer introduced in
§19.1/Appendix B (Q3); and analyze the memory-access pattern of the
GEMM-formulation kernel (Fig. 19.11), discussing whether its global
accesses are coalesced and whether its shared-memory accesses may
suffer bank conflicts (Q4). Not implemented in this repo's samples
(end-of-chapter exercises are out of scope — see the repo root README)
— though Q4's bank-conflict/coalescing analysis overlaps with analysis
work already captured in this chapter's README.
