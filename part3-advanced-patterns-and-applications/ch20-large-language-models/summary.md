# Chapter 20 — Large Language Models

Section-by-section summary of the book content. Page numbers are printed
book pages (Chapter 20 runs pp. 477–511).

Statistical language modeling predates deep learning (IBM's work),
but 2017's **Transformer** architecture and 2018's BERT made it the
backbone of modern NLP. LLMs distinguish themselves from earlier
language models mainly by scale (orders of magnitude more parameters/
training data); multi-modal models (vision, audio, etc.) are a further
generalization, briefly noted but out of scope. An LLM's core task is
predicting the next **token** given the preceding **context**
(system prompt + user input + tokens already generated). Accurate
generation over long contexts requires **attention** — a mechanism
that identifies which parts of the context matter most for generating
the next token — making attention and the Transformer architecture the
chapter's central subject.

## 20.1 Transformer architecture (p. 478)

Attention resembles an in-network classification process using
**softmax** to produce probabilities. The original Transformer (for
machine translation) had two layer types: **encoder** layers
(contextualize input tokens into embedding vectors) and **decoder**
layers (generate output tokens one at a time, auto-regressively,
conditioned on both encoder output and previously-generated output
tokens). Later adaptations drop whichever layer type a task doesn't
need: **discriminative** tasks (e.g. classification) need no decoder;
**generative** tasks (e.g. LLM text generation) are commonly
**decoder-only**.

Fig. 20.1's decoder-only architecture, component by component: (1)
**Tokenizer** converts text into integer token IDs, built from a
vocabulary learned during training (each novel text fragment gets a
newly assigned ID, recorded in a fragment→ID mapping for reuse). (2)
**Embedding** layer looks up each token ID's floating-point vector
representation (also learned during training) — needed since
gradient-descent training requires differentiable, continuous-valued
functions throughout. (3) **Positional encoding** adds sequence-order
information to embeddings, informing attention of tokens' relative
physical position/proximity. The accumulated embeddings of the whole
conversation-so-far form matrix `X` (one row per token), the first
transformer layer's input. (4) Each **Transformer layer** contains: (5)
a **multi-head attention** sub-layer — a parallel array of *attention
heads*, each linearly projecting `X` then comparing projected tokens
for similarity (detailed in §20.2); outputs from all heads are
concatenated and linearly transformed into one combined output. (7) An
**Addition & Normalization** ("Add & Norm") sub-layer follows both the
attention and feed-forward sub-layers: addition enforces a **residual
connection** (retaining information from the sub-layer's original
input), and normalization (using input mean/std plus learned
parameters γ/β) stabilizes training convergence. (6) A **Feed-forward**
sub-layer (two linear transformations with a ReLU in between) adds
semantic meaning to the similarity-weighted token representations the
attention sub-layer produced. (8) A final **Un-embedding** layer
converts the last transformer layer's output embeddings into a
probability distribution over the vocabulary, used to generate the
next token.

Three model dimensions determine parameter count and capability:
**model depth** (number of transformer layers — deeper models capture
more general dependency patterns), **head dimension** `d` (elements
per embedding vector — wider captures more semantic nuance), and
**number of heads** (captures different dependency *types* per
sub-layer); total parameters scale with the product of these three,
reaching billions/trillions in practice. Since user inputs are often
too terse for accurate generation despite needing long context,
**retrieval-augmented generation (RAG)** concatenates retrieved
relevant documents with the user's input to enrich the context.
Attention heads' large-scale matrix multiplications (both training and
inference) are the most computationally expensive component of LLMs,
and the chapter's focus for the rest of the chapter.

## 20.2 Multi-head attention (p. 482)

Each attention sub-layer's multiple heads specialize (during training)
to capture different dependency types — positional, semantic,
syntactic — via independently-initialized weight matrices, with
gradient descent naturally avoiding head overlap, aided by dropout and
layer normalization. Heads are mutually independent and so are
computed concurrently (e.g. across different thread blocks); the
chapter details the design of one head.

Each head's input `X` is an `N×d` matrix (`N` = **sequence/context
length**, the number of tokens generated so far; `d` = head dimension).
Attention sub-layers typically enforce a **causality policy**: token
generation depends only on earlier tokens, never future ones.

**Step 1**: linear projection generates **query (Q)**, **key (K)**,
and **value (V)** matrices via learned `d×d` weight matrices
`W_Q`/`W_K`/`W_V` (`Q=X·W_Q`, etc. — Fig. 20.2 left). **Step 2** (Eq.
20.1, Fig. 20.2 right): `O = softmax(QKᵀ/√d + M)·V`. `QKᵀ` is `N×N`
(called `S`); element `(i,j)` is an inner product between projected
embeddings for tokens `i` and `j` — mathematically a **cosine
similarity** (given unit-length, normalized vectors), so `S` is a
pairwise similarity map between all tokens. The `1/√d` **scaling
factor** prevents dot products from growing too large with large head
dimensions, which would otherwise saturate softmax and cause vanishing
gradients during training. The **mask matrix M** adds `-∞` to every
element of `S` whose column index exceeds its row index, enforcing
causality. **Softmax** (Eq. 20.2) converts each row of `QKᵀ+M`
(treated as logits `l_{r,c}`) into a probability distribution
`p_{r,c}`; subtracting each row's max logit `m_r` before exponentiating
(mathematically a no-op, since it multiplies numerator and denominator
by the same `e^{-m_r}`) prevents floating-point overflow. With the
causality mask, any logit with `row < column` becomes `e^{-∞}=0` after
softmax, making `P=softmax(QKᵀ+M)` a **lower-triangular** probability
matrix. The final step multiplies `P` by `V`: each token's new
embedding (row of output `O`) is a similarity-probability-weighted sum
of the `V` embeddings of all tokens generated at or before it.

**Inference stages**: LLM inference with this decoder-only
architecture proceeds in two phases (Fig. 20.3). **Summarization**
(the first pass): the system prompt, user input, and any RAG-augmented
tokens are concatenated and embedded into the initial `X` matrix for
the first transformer layer; this propagates through every transformer
layer and the un-embedding layer to produce the *first* output token.
**Generation** (a.k.a. **decoding**, the **autoregression phase**):
each subsequent output token becomes a new input row, re-triggering
the full stack of transformer layers to produce the *next* output
token — iterating until an end-of-sequence (EOS) token is generated.

## 20.3 Implementing attention in CUDA (p. 486)

Attention computation is two matrix multiplications (`QKᵀ` and `P·V`),
one elementwise scaling, one elementwise masked addition, and one
softmax — the matrix multiplications can use custom kernels or
optimized library GEMM calls (e.g. cuBLAS); the only genuinely new
piece is **softmax** itself, which the chapter implements explicitly
(fusing the causality mask addition into the same kernel, to avoid an
extra pass over `S`, improving arithmetic intensity and GPU
utilization).

Fig. 20.4's `softmax_kernel`: a 1D grid of 1D thread blocks, one block
per row of `S` (`BLOCK_SIZE` is a compile-time-constant,
occupancy-tuned block width, chosen to minimize per-row iterations
while maximizing SM occupancy). Each thread loop's exit condition
(`idx <= blockIdx.x`) implicitly applies the causality mask — simply
*skipping* elements past the diagonal, rather than literally adding
`-∞` to them. The kernel computes each row's max (lines 6-14, via a
thread-coarsened reduction using CUB's `BlockReduce`, echoing Chapter
10) and sum-of-exponentials denominator (lines 15-23, same pattern),
sharing each result across the block via shared memory plus a
`__syncthreads()`, then (lines 24-25) writes every row element's final
softmax probability into `P` — with elements past the diagonal
explicitly set to 0, per the causality policy. The denominator (`D`)
is also stored to global memory for reuse during training. No
global/grid-wide barrier is needed — only a block-level
`__syncthreads()` — since each thread block independently owns one
complete row. The matrix multiplications for generating `S` and `O`
are left as an exercise.

## 20.4 KV caching (p. 488)

Naive re-computation (Fig. 20.3) is wasteful: each generation-phase
iteration recomputes **full** `N×d` matrix multiplications even though
only **one new row** of `X` is added per iteration (the newly-generated
token). Fig. 20.5 shows which parts of `Q`, `K`, `V`, and `QKᵀ` actually
*change*: `Q`, `K`, `V` each only gain one new row (via a cheap
vector-matrix multiplication of just the new `X` row against
`W_Q`/`W_K`/`W_V`); the `QKᵀ` product matrix is **unchanged** at every
`(r,c)` with both `r,c < N` (since those values only ever derived from
rows of `Q`/`K` that haven't changed) — only its new final row and
column actually need computing (and the new column is all-zero except
its last element, per the causality mask). Softmax's result is
similarly unaffected for existing rows, since the new, zero-valued
column entries don't dilute previously-computed probabilities.

Since every iteration needs the **entire** `K` and `V` matrices (not
just their new rows) to compute attention, these are **memoized** —
stored and reused — in the so-called **KV cache** (Fig. 20.6 details
the vectors/matrices involved; distinct from "key-value store" in the
general data-structures sense, though analogous). Fig. 20.7 revises
the inference-stage picture with KV caching: the summarization phase
becomes the **prefill** phase (initial `K`/`V` computed and cached for
every layer); each generation iteration computes only new rows
`Q'`/`K'`/`V'` via cheap vector-matrix multiplications, appends `K'`/`V'`
to each layer's cache, and uses the full cached `K`/`V` to compute
`O'`. The prefill phase still needs one full-sequence transformer
pass (large GEMMs, **compute-bound**, high GPU utilization) — §20.5's
flash attention targets exactly this phase. Each generation-phase pass,
by contrast, performs mostly **vector-matrix** multiplications (GEMVs)
— **memory-bandwidth-bound**, low arithmetic intensity, typically
under-utilizing GPU compute — addressed in §20.6.

## 20.5 Flash attention (p. 492)

A naive multi-kernel softmax implementation (as in Fig. 20.4) forces
**global barriers** at kernel boundaries plus repeated global-memory
reads/writes of entire intermediate matrices (`S`, `P`) — a major
bottleneck. **Flash attention** mathematically reformulates attention
so its operations can be **fused into a single kernel** and tiled, so
intermediate matrices never touch global memory at all.

The key obstacle to naive tiling: softmax (Eq. 20.2) needs each row's
**max** and **sum-of-exponentials** before any output elements of that
row can be finalized — seemingly requiring two full passes over every
row. The fix is an **online, composable partial-softmax** formulation:
define `D_{r,A}` (Eq. 20.3) as the partial exponential-sum over just a
*subset* `A` of a row's columns, with a **composition rule** (Eq. 20.4)
showing how to merge two disjoint subsets' partial sums (and running
maxima) into the sum for their union — letting partial results from
different tiles be computed independently (in any order, even in
parallel) and merged incrementally. The same composability is extended
to the **output** matrix `O` (Eqs. 20.5–20.6): a tile's partial
contribution to `O` can be computed from its own local max/sum, then
**rescaled** and merged into the running output total as later tiles'
contributions arrive — avoiding ever materializing the full `S`/`P`
matrices.

Fig. 20.8 illustrates the resulting tiling scheme: each thread block
owns a horizontal panel of `Q` (`B_r` rows) and the full `Kᵀ`/`V`, and
iterates through `Kᵀ`/`V` in `B_c`-wide tiles, incrementally
accumulating its contribution to the corresponding panel of `O` — all
intermediate data (`S`/`P` tiles, running max `m_i`, running
denominator `D_i`) stays in shared memory/registers the whole time, no
global synchronization needed between tiles. Flash attention is an
**exact** reformulation (mathematically identical results, not an
approximation), and applies to both training (forward and backward
pass — the backward pass needs 5 matrix multiplications vs. the
forward pass's 2, and is conceptually simpler since it needs no
softmax rescaling) and inference; the chapter covers only the forward
pass.

Fig. 20.9's complete kernel (`flashattention_forward_kernel`),
implementing one attention head's forward pass: a 1D grid of 1D thread
blocks, each generating one horizontal panel of `O`; within a block,
work is split **warp-level**, each warp owning a `B_r_warp × d`
sub-panel of `Q`/`O`. Supporting device functions: `load_Q()` (Fig.
20.10, interleaved coalesced load into registers); `load_KT_and_V()`
(Fig. 20.11, loads one `Kᵀ`/`V` tile into shared memory each
block-level iteration — `Kᵀ` loaded transposed and **padded** per-column
to avoid shared-memory bank conflicts, reusing the `addr()` padding
trick from Chapter 16); `compute_S_and_max()` (Fig. 20.12, a
warp-level vector-matrix multiplication between `Q`'s sub-panel and
the current `Kᵀ` tile, applying the causality mask and scaling factor,
then a CUB `WarpReduce` to get the row's running max, broadcast back
via `__shfl_sync`); `update_m_and_D()` (Fig. 20.14, rescales the
running max/denominator per Eq. 20.4 as a new tile's contribution is
merged in — **note**: the book's own prose describing this figure's
line 6 attributes the computed term to `D_{r,B}·e^{m_{r,B}−m_{r,A∪B}}`,
but by the code's own variable assignments (`D` initialized from the
*previous* `D_i[ii]`, i.e. `D_{r,A}`, and `last_m` from the *previous*
`m_i[ii]`, i.e. `m_{r,A}`) line 6 actually computes the *other* term of
Eq. 20.4, `D_{r,A}·e^{m_{r,A}−m_{r,A∪B}}` — a genuine subscript
mismatch in the published text, not an OCR artifact, confirmed by
reading the rendered page directly); `compute_P_and_update_D()` (Fig.
20.15, applies softmax's exponential step to the current `S` tile in
place as `P`, accumulating into the running denominator); and
`compute_O()` (Fig. 20.13, rescales the running `O` sub-panel per the
merge rule, then accumulates the current tile's vector-matrix
contribution from `P` and `V`). `store_O()` (Fig. 20.16) writes the
final (denominator-divided) output panel and raw denominator vector to
global memory at the very end — the only global writes the whole
kernel performs for its output.

The most recent flash attention version additionally exploits
Hopper-specific hardware: the **Tensor Memory Accelerator (TMA)**
(asynchronous global↔shared memory tile movement with no compute-thread
involvement), asynchronous **WGMMA** tensor-core instructions, and FP8
tensor-core precision — using warp specialization to overlap data
movement, computation, and (asynchronous) softmax more aggressively.

## 20.6 KV cache arithmetic intensity and memory requirement (p. 504)

The generation phase's per-token `K`/`V` row computations are
vector-matrix operations with inherently **low arithmetic intensity**,
under-utilizing GPU compute (Chapter 5) — motivating **batching** and
**speculative decoding** as ways to raise it.

**Batching** (processing multiple independent sequences together) is
a standard DNN technique for raising arithmetic intensity, since
batching reuses the *same weights* across more input vectors (worked
example: a 1024-in/4096-out linear layer's AI rises from 1 FLOP/B at
batch size 1 to 315 FLOP/B at batch size 512). But in LLM inference
this benefit applies **only to the linear-projection layers**
(`Q`/`K`/`V` generation, which share weights across users) — attention
itself does **not** benefit from batching, since each user's KV cache
is tied to their own distinct prompt/context (Fig. 20.17) and so can't
be shared the way weights can.

**Memory requirements**: total LLM inference memory is dominated by
model weights (e.g. a 7B-parameter model in FP16 ≈ 14 GB) plus the
**KV cache**, whose per-token size for multi-head attention (MHA) is
`2·l·h_q·d·p` (Eq. 20.7 — `l`=layers, `h_q`=query heads, `d`=head
dimension, `p`=bytes per value, factor 2 for both K and V); across a
batch of `b` sequences of length `N`, total size scales as
`b·N·2·l·h_q·d·p` (Eq. 20.8). Worked real-model examples for one
4096-token sequence (16-bit): PaLM 2 ≈ 8 GB, GPT-3 ≈ 18 GB, Llama 2 7B
≈ 2 GB — sizes large enough that batching multiple such sequences
commonly exceeds a single GPU's memory, motivating multi-GPU/multi-node
serving (Chapter 23). Real-world batches also have **varying sequence
lengths** across users, risking load imbalance across thread blocks;
production systems address this with **in-flight batching** (scheduling
new requests into whichever batch currently has the lowest workload,
as soon as a slot frees up).

Estimating attention's own arithmetic intensity (ignoring softmax,
`p=1`): per layer/token, KV cache data moved is `2·h_q·d·N`, `Q'`/`O'`
movement is `2·h_q·d`, and FLOPs for `(Q'Kᵀ)V` are `2·N·h_q·d` — giving
`AI_MHA ≈ N/(1+N) ≈ 1` FLOP/byte (Eq. 20.9) — confirming attention
stays deeply memory-bound regardless of batch size.

**Speculative decoding** (Fig. 20.18) instead uses a small, fast
**draft model** to predict several (a *speculation depth* of) candidate
future tokens, which the large **target model** then verifies **in
parallel** against its own predictions — accepting the longest
correctly-matching prefix and only discarding speculation past the
first mismatch. Because all speculated continuations share the same
*existing* KV cache content, this amounts to a special form of batching
where the shared KV cache data is loaded and reused once across
multiple candidate continuations (rather than reloaded once per output
token), directly raising arithmetic intensity.

## 20.7 Alleviating the memory requirements of the attention mechanism (p. 508)

MHA's large KV cache stems from each head using its *own* `K`/`V`
weight matrices. **Multi-query attention (MQA)** shares a single `K`/`V`
pair across **all** heads (Fig. 20.19b), cutting KV cache size by a
factor of `h_q` (Eq. 20.10) and raising arithmetic intensity by
approximately `h_q` (Eq. 20.11) — at some cost to model accuracy, since
all heads now share identical key/value representations. **Grouped-
query attention (GQA)** balances MHA and MQA by sharing `K`/`V` across
**groups** of `g_q` heads each (Fig. 20.19c), reducing memory by a
factor of `g_q` (Eq. 20.12) and raising AI by approximately `g_q`
(Eq. 20.13) — a tunable middle ground. All three (MHA, MQA, GQA) can
use the same flash-attention tiling approach from §20.5; the choice
trades off compute, memory, and accuracy.

Even with MQA/GQA, over-provisioned (worst-case-sized) KV cache memory
tends to be fragmented and wasted across a batch of variously-sized
requests. **PagedAttention**, inspired by OS virtual-memory paging,
partitions the KV cache into equal-size blocks loaded from host memory
on demand, reducing this fragmentation. **Multi-head Latent Attention
(MLA)** (Fig. 20.19d) takes a different approach: it compresses each
token's `K`/`V` into a shared, low-rank **latent vector** (learned
projection matrices), storing only this compact latent representation
in the KV cache (Eq. 20.14 — note the complete absence of `h_q` from
the formula, versus MHA/MQA/GQA) and **decompressing** it (via learned
up-projection matrices) back into per-head `K`/`V` only when actually
needed during a generation step — raising arithmetic intensity by
roughly `2·h_q` (Eq. 20.15), a larger factor than MQA/GQA typically
achieve. A different, orthogonal scaling technique, **Mixture of
Experts (MoE)**, is named briefly: a gating mechanism activates only a
small subset of specialized sub-networks per input, allowing very
large total parameter counts with much lower per-token compute.

## 20.8 Summary (p. 510)

The chapter introduced the computational operations underlying LLMs,
focused on the Transformer's distinguishing component: attention
heads. Starting from a naive multi-kernel implementation (emphasizing
the softmax kernel as the genuinely novel piece), it introduced **KV
caching** to eliminate redundant recomputation during generation, then
**flash attention** to fuse and tile the prefill phase's computation
for much higher arithmetic intensity and lower memory traffic. The
chapter closed by analyzing KV cache memory requirements and surveying
techniques (batching, speculative decoding, MQA/GQA, PagedAttention,
MLA) that alleviate the resulting memory and utilization pressures
during inference — equipping the reader with the fundamentals of LLM
implementation on GPUs, and a large further design space to explore.

## 20.9 Exercises (p. 511)

Four problems: complete the host code to invoke the flash attention
kernel of Fig. 20.9, sizing input/output arrays per Fig. 20.8's
dimensions (Q1); study the CUB `WarpReduce` documentation's data types
and `Reduce` argument semantics, including its range of supported
arithmetic functions (Q2); implement the initialization device
function referenced at line 21 of Fig. 20.9, explaining why `O_i` and
`D_i` need their particular initial values (Q3); and prove that all
loads from `K` and `V` in `load_KT_and_V()`'s nested loop (Fig. 20.11)
are coalesced (Q4). Not implemented in this repo's samples
(end-of-chapter exercises are out of scope — see the repo root
README).
