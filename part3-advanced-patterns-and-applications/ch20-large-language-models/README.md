# Chapter 20: Large language models

Source: *Programming Massively Parallel Processors*, 5th ed., Ch. 20 (pp. 477-511).

| File | Book section | Technique |
|------|-------------|-----------|
| `01_attention_naive.cu` | §20.2-20.3, Eq. (20.1)-(20.2), Fig. 20.4 | Naive single-head scaled dot-product attention: separate `QK^T`, causal-softmax, and `PV` kernels; the causal-softmax kernel matches Fig. 20.4's one-block-per-row, thread-coarsened CUB `BlockReduce` design exactly |
| `02_attention_kv_cache.cu` | §20.4, Figs. 20.5-20.7 | Incremental (autoregressive) attention with a growing KV cache: prefill fills the cache directly, then each generation step appends one new K/V row and computes only the new row of `S`/`P`/`O` via GEMV against the full cache |
| `03_flash_attention.cu` | §20.5, Eq. (20.3)-(20.6), Figs. 20.8-20.16 | Tiled, online-softmax flash attention: fuses `QK^T`, causal-softmax, and `PV` into one kernel using the online-softmax composition rule, never materializing the full `N x N` attention matrix |

Each file is fully self-contained (own device functions, own CPU reference,
own random test-input generation -- no includes between the three files,
per this repo's convention). All three use the same problem size for
consistency: sequence length `N=64`, head dimension `d=32` (single head,
`d == WARP_SIZE`), a causal (autoregressive) mask, and scale `1/sqrt(d)`
applied to `QK^T` per Eq. (20.1). Every file generates its own random
`Q`/`K`/`V`, computes a full causal-softmax attention reference in **double
precision**, runs its CUDA kernel(s), compares with `nearlyEqual`, and
prints `PASS`/`FAIL` plus a timing line.

**A note on epsilon.** The repo's default `nearlyEqual` epsilon is `1e-3`.
All three files here use a slightly looser epsilon because softmax's
exp/sum/normalize chain accumulates floating-point rounding differently on
the GPU (values combined via a parallel tree reduction -- CUB's
`BlockReduce`/`WarpReduce`) than in the CPU reference's sequential
double-precision accumulation, and file 03 additionally applies an
`expf()`-based rescale to the running output/denominator at every one of
its `T_c` tile merges, compounding one more rounding step per merge. Files
01 and 02 use `eps=2e-3`; file 03 uses `eps=3e-3`. In practice, measured
`max|diff|` against the double-precision reference is ~1e-7 for all three
files at this problem size, so these epsilons leave ample headroom while
still being meaningfully tighter than a no-op bound.

## §20.2-20.3 Naive attention -- `01_attention_naive.cu`

Implements Eq. (20.1), `O = softmax(QK^T/sqrt(d) + M)V`, as three kernels.
§20.3 explicitly scopes the chapter's CUDA-implementation focus to the
softmax kernel ("the only new aspect of attention computation is the
implementation of the softmax function... We will thus focus on the CUDA
implementation of the softmax function and leave the rest as an exercise"),
so the two matrix multiplications (`QK^T` and `PV`) are simple
one-thread-per-output-element kernels, while the causal-softmax kernel
(`softmaxCausalKernel`) matches Fig. 20.4 precisely:

- a 1D grid of `N` thread blocks, one block per row of `S` (§20.3:
  "`gridDim.x` ... is set to the number of rows in matrix S");
- `BLOCK_SIZE` threads per block collaboratively reducing over their row;
- causality (Eq. 20.1's mask `M`) is enforced without ever materializing
  `M`, by bounding the reduction/output loops at `idx <= blockIdx.x`
  (§20.3: "the exit condition of the for loop ... is `idx <= blockIdx.x`
  ... which implements the causality policy without explicitly adding
  matrix M");
- the row maximum `m_r` and the softmax denominator `D_r` (Eq. 20.2) are
  each computed by a thread-coarsened reduction -- a private partial value
  per thread, a block-wide CUB `BlockReduce`, then a broadcast through a
  shared-memory scalar -- exactly Fig. 20.4's two-pass structure;
- the denominator is additionally written to its own output vector `D`
  (Fig. 20.4 line 27), "for reuse during training", even though this
  inference-only sample does not otherwise use it.

## §20.4 KV caching -- `02_attention_kv_cache.cu`

§20.4 observes that from one decoding iteration to the next, `X` gains
exactly one new row, so `Q`, `K`, `V` each only need one new row (Fig.
20.6), all elements of `QK^T` with both indices `< N` are unchanged, the
new row of `QK^T` is a vector-matrix product against the full (cached) `K`,
and (per the causality policy) all rows of `O` below the new one are
unchanged -- "one simply needs to perform a vector-matrix multiplication
between the new row of `softmax(QK^T)` and the new V". §20.4: "this
requirement can be met by memoizing K and V ... in the form of the
so-called KV cache."

This file reproduces both phases of Fig. 20.7 on synthetic per-token `Q`,
`K`, `V` rows (the trivial, un-discussed vector-matrix projections against
`W_Q`/`W_K`/`W_V` that would normally produce these rows are outside this
chapter's scope -- §20.4's actual subject is what happens to `K`, `V`, and
`O` once the rows exist):

- **prefill** (summarization): the first `N_PREFILL=8` tokens' `K`, `V`
  rows fill the KV cache directly;
- **generation** (decoding): for each token `i = N_PREFILL..N-1`, its new
  `K_i`/`V_i` row is appended to the cache, and only the new row of `S`,
  `P`, `O` is computed as a GEMV against the *entire* cached `K^T`/`V` --
  never a full `N x N` recompute. Causality holds automatically for the new
  row with no explicit masking (its valid range `[0, i]` is exactly what
  the cache holds so far), exactly as §20.4 describes.

At every generation step, the incrementally-produced `O_i` row is checked
against a full causal-softmax recompute over the first `i+1` tokens (the
same double-precision reference formulation as file 01, reproduced locally)
-- verifying the KV-cache shortcut is numerically equivalent to full
recomputation at every single step, not just at the end.

## §20.5 Flash attention -- `03_flash_attention.cu`

Implements the forward-pass kernel of Fig. 20.9: fuses `QK^T`, causal
softmax, and `PV` into one kernel using the tiled, online-softmax
composition rule of Eq. (20.3)-(20.6), so the full `N x N` `S`/`P` matrices
are never materialized in global memory.

**Tiling scheme (Fig. 20.8).** Each thread block owns a horizontal panel of
`B_r=16` contiguous rows of `Q`/`O`. The panel is split into `B_r_warp`-row
sub-panels, one per warp; this sample uses `WARPS_PER_BLOCK=1`, the
simplest valid case, so `B_r_warp == B_r` and every "warp-level" step is
simply the block's one warp. The block iterates `T_c = N/B_c = 2` times
over `B_c=32`-column tiles of `K^T` and `B_c`-row tiles of `V` (held in
shared memory), merging each tile's contribution into the running `O`
panel. `d=D_HEAD=32 == WARP_SIZE`, the simplest valid case of §20.5's "we
assume that d is a multiple of WARP_SIZE"; `B_c == WARP_SIZE` too, per
§20.5's "we only allow B_c value to be multiples of WARP_SIZE".

**Online-softmax recurrence (Eq. 20.4/20.6).** For row `r` and column
subsets `A` (already merged) and `B` (the new tile), with `m_r,X`/`D_r,X`
the running max/denominator restricted to subset `X`:

```
m_r,A∪B   = max(m_r,A, m_r,B)
D_r,A∪B   = D_r,A * e^(m_r,A - m_r,A∪B) + D_r,B * e^(m_r,B - m_r,A∪B)
o_r,c,A∪B = o_r,c,A * e^(m_r,A - m_r,A∪B) + o_r,c,B * e^(m_r,B - m_r,A∪B)
```

The kernel evaluates the new tile's (`B`-subset) `P`/`O` contributions
directly relative to the already-updated merged max `m_r,A∪B` (Fig. 20.15's
exponentials use the final `m_i`, and per the book, "there is no need to
rescale these P elements when merging them into O"), so only the *old*
accumulated `D_i`/`O_i` need the `e^(m_r,A - m_r,A∪B)` rescale before the
new terms are added in -- `updateMAndD()` rescales `D_i`, `computeO()`
rescales `O_i`, matching Figs. 20.14 and 20.13 respectively.

**A note on a probable OCR artifact in the extracted chapter text.** The
prose describing `update_m_and_D()` (Fig. 20.14) states that its line 6
"computes the term `D_r,B * e^(m_r,B - m_r,A∪B)`" -- but at that point `D_i`
still holds only the old `D_r,A`; the new tile's contribution `D_r,B` isn't
summed until `compute_P_and_update_D()` runs afterwards, and
`update_m_and_D()` has no access to a not-yet-computed `D_r,B`. The only
interpretation consistent with the code's actual data flow -- and the one
implemented here -- is that this line rescales the *old* term, i.e.
computes `D_i = D_r,A * e^(m_r,A - m_r,A∪B)`, the first addend of Eq.
(20.4)/(20.6); `compute_P_and_update_D()` then computes the `D_r,B` term
directly against the merged max and adds it, completing the composition
rule with no separate rescale needed for that term. This reads as a
subscript transcription slip (A vs. B) in the source PDF's math rendering,
not a deviation this file takes from the book -- the implementation follows
Eq. (20.4)/(20.6) exactly, and is verified correct against the naive
double-precision reference.

**Other fidelity notes.** `KT_j` is padded in shared memory
(`addr(x) = x + (x >> LOG_NUM_BANKS)`) to avoid bank conflicts, per §20.5;
`V_j` is not padded, matching the book. Q rows are loaded into registers
interleaved across the warp via `__shfl_sync` broadcasts (Fig. 20.10), and
`compute_S_and_max`/`compute_O` (Figs. 20.12-20.13) perform their
vector-matrix products the same way. A `__syncwarp()` was added between
`compute_P_and_update_D()`'s shared-memory write and `compute_O()`'s
cross-lane read of the same tile (every lane reads every other lane's `P`
value there) -- required because independent thread scheduling (Volta+)
does not guarantee same-warp shared-memory writes are visible without an
explicit reconvergence point; `compute-sanitizer --tool racecheck` flagged
this exact hazard without it and is clean with it.

## §20.7: memory-alleviation techniques (no separate sample)

§20.7 presents multi-query attention (MQA), grouped-query attention (GQA),
PagedAttention, and multi-head latent attention (MLA) purely as memory/
arithmetic-intensity analyses (Eq. 20.10-20.15) over the KV-cache-size
formula from §20.6 (Eq. 20.7-20.9) -- there is no CUDA code listing in this
section, only algebraic reductions in how much of `K`/`V` gets cached per
token:

- **MQA** shares one K/V per layer across all query heads (Eq. 20.10),
  raising the arithmetic intensity of attention by roughly `h_q` (the
  number of query heads, Eq. 20.11) at the cost of the reduced KV
  expressiveness (one shared K/V instead of `h_q` distinct ones).
- **GQA** [15] splits the difference: `g_q` groups of query heads each
  share one K/V (Eq. 20.12), an arithmetic-intensity gain of roughly `g_q`
  (Eq. 20.13), tunable between MHA (`g_q = h_q`) and MQA (`g_q = 1`).
- **PagedAttention** [12] partitions the KV cache into fixed-size blocks
  loaded on demand, borrowing OS-style paging to reduce the fragmentation
  caused by over-provisioning KV-cache memory for the worst-case sequence
  length.
- **Multi-head latent attention (MLA)** [16] compresses K and V per token
  into one shared low-rank latent vector across all heads and layers (Eq.
  20.14), decompressed via up-projection matrices only when a head is
  active, for the largest memory reduction of the four (arithmetic
  intensity roughly `2 * h_q`, Eq. 20.15).

Per the task's scope rule (a fourth file is added only if the chapter text
contains an actual code listing for §20.7), no sample is included for this
section -- it is a set of memory/bandwidth tradeoffs layered on top of the
same flash-attention-tileable computation already implemented in file 03,
not a new kernel.

## Results

```
== bin/01_attention_naive ==
Naive single-head scaled dot-product attention (§20.2-20.3, Eq. 20.1-20.2, Fig. 20.4):
N=64, d=32, causal mask, softmax eps=0.0020
GPU vs CPU-double reference: max|diff|=0.000000  0.1065 ms  [match]
PASS
== bin/02_attention_kv_cache ==
Incremental attention with a growing KV cache (§20.4, Figs. 20.5-20.7):
N=64, prefill=8, d=32, softmax eps=0.0020
generation steps 8..63: max|diff|=0.000000, total kernel time=8.8950 ms  [match]
PASS
== bin/03_flash_attention ==
Tiled, online-softmax flash attention forward pass (§20.5, Eq. 20.3-20.6, Fig. 20.9):
N=64, d=32, B_r=16, B_c=32, warps/block=1, T_r=4, T_c=2, softmax eps=0.0030
GPU flash attention vs CPU-double naive reference: max|diff|=0.000000  3.6380 ms  [match]
PASS
```

All three binaries also pass under `compute-sanitizer --tool memcheck`,
`--tool racecheck`, and `--tool synccheck` (0 errors/warnings) at
`-arch=sm_75`.

Build and run all samples in this chapter:

```sh
make run
```
