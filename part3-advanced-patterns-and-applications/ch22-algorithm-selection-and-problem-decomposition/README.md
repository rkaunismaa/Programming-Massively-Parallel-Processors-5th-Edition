# Chapter 22: Algorithm selection, problem decomposition, and problem formulation

Source: *Programming Massively Parallel Processors*, 5th ed., Ch. 22 (pp. 529-540).

| File | Book section | Technique |
|------|-------------|-----------|
| `01_batching_latency_vs_throughput.cu` | §22.5 | Batching queries into one matrix multiply vs. processing each with its own vector-matrix multiply: per-query latency vs. total-latency/throughput trade-off |

## Why this chapter is different

Unlike the kernel-by-kernel chapters, Chapter 22 is a reflective,
design-methodology chapter. Its four content sections do the following:

- **§22.1 Algorithm selection** reviews, purely in prose, the
  complexity-vs-parallelism trade-off already implemented in this repo:
  Kogge-Stone vs. Brent-Kung prefix sum (Ch. 11), odd-even/merge/radix sort
  (Ch. 14), and DCS vs. cutoff-summation (Ch. 21). It introduces no new
  algorithm or code -- it is a retrospective.
- **§22.2 Problem decomposition** defines output-centric (gather) vs.
  input-centric (scatter) decomposition and then walks through essentially
  every pattern earlier in the book (image processing, matmul, convolution,
  stencil, histogram, reduction, scan, filters, merge, sort, wavefront,
  SpMV formats, graph traversal, electrostatic potential map) classifying
  each as one or the other. Fig. 22.1 is a two-panel diagram illustrating
  gather vs. scatter access patterns abstractly (not a code figure). No new
  kernel is introduced; every kernel discussed already exists in this repo
  under its own chapter.
- **§22.3 Amdahl's law** is a single fully worked arithmetic example (a
  molecular-dynamics application where a 95%-of-runtime, 100x-accelerated
  module still caps the whole application at ~17-20x speedup) -- the
  calculation is given inline as arithmetic in the prose
  (`1/(5% + 95%/100) = 17x`, `1/5% = 20x`), not as an algorithm to
  implement. Fig. 22.2 is an application block diagram, not code.
- **§22.4 Problem formulation** is a half-page discussion citing the Ch. 21
  cutoff-binning trade-off (already implemented as `05_dcs_cutoff_binning.cu`
  under Chapter 21) as an example of rethinking a numerical method rather
  than mechanically translating one; it presents no new computation.

None of §22.1-22.4 contains a distinct, implementable listing: every
concrete kernel they reference is prose-level recall of a kernel already
implemented elsewhere in this repo (Chapters 3, 5, 7-14, 16-18, 21), and
implementing them again here would not be grounded in anything *this*
chapter presents -- it would just be re-implementing another chapter's
sample under a different directory.

## §22.5 is the one exception -- what was implemented and why

§22.5 "Batching: latency vs. throughput" describes a concrete, self-contained
computational comparison that is *not* an implementation elsewhere in the
book: a QKV projection is a vector-matrix multiplication per query; batching
many queries together turns the group of vector-matrix multiplications into
one matrix multiplication, which "takes much shorter time than the sum of
the execution time for all the vector-matrix multiplications in the batch,"
at the cost of any individual query's own latency (it must now wait for the
whole batch). The section gives the throughput formulas explicitly:

```
throughput_batched   = total_queries / latency_for_one_matrix_multiplication
throughput_unbatched = total_queries / total_latency_for_all_vector_matrix_multiplications
```

**Important caveat:** the book gives no code listing or figure for this --
it is a prose description with inline formulas only (there is no "Fig. 22.x"
anywhere near §22.5). `01_batching_latency_vs_throughput.cu` is therefore a
*faithful reconstruction* of the described comparison, built from standard
techniques already used elsewhere in this repo (a naive one-thread-per-output
vector-matrix-multiply kernel, and Chapter 5's shared-memory tiled matmul
technique adapted to an `A * B^T` access pattern matching the real
`[out_features][in_features]` weight layout used for Q/K/V projections in
Chapter 20's attention samples) -- not a transcription of anything printed
in the book. This is a judgment call: the alternative reading is that §22.5
is also "purely conceptual" and should get no code. It was implemented
because, unlike §22.1-22.4, the comparison it describes is precise enough
(explicit formulas, a concrete pair of computations being contrasted) to
implement without inventing unstated details, and the task brief's own
Step 2 names exactly this construction as the kind of thing worth
implementing for this chapter.

### What the sample does

- `d = 256` features, `B = 2048` independent "queries" (random vectors).
- **Unbatched path**: `matvecKernel` launched once per query (2048 separate
  kernel launches), each a naive one-thread-per-output-feature
  vector-matrix multiply with no reuse of the weight matrix across queries.
- **Batched path**: `batchedProjectionKernel`, a single shared-memory tiled
  matmul launch (`Y = X * W^T`) computing all 2048 outputs at once, reusing
  tiles of the weight matrix across many queries.
- A double-precision CPU reference checks both GPU paths for correctness
  (`nearlyEqual`), and the two GPU paths are cross-checked against each
  other.
- Timing (via `GpuTimer`, CUDA events) reports: latency of a single lone
  query (unbatched), total latency to finish all 2048 queries unbatched,
  and the batched kernel's single-launch latency (which is *both* the
  total latency for all 2048 queries *and* the per-query latency any one
  of them experiences, since none of them finish before the others).

### Measured result (this machine, optimized build, `DEBUG=0`)

```
Single-query latency (unbatched, one launch):        0.0195 ms
Total latency, unbatched, all 2048 queries:          30.9852 ms  (9,634-66,096 queries/s across runs)
Total/per-query latency, batched, all 2048 queries:   0.0543 ms  (throughput = 3.7e7 queries/s)
Throughput gain from batching: ~340-570x (varies by build/run)
```

This reproduces the book's qualitative claim exactly: the batched path's
per-query latency (0.054 ms) is higher than a lone unbatched query's latency
(0.020 ms), yet the batched path's *total* latency for the whole group is
far lower than summing 2048 individual unbatched launches, so throughput is
dramatically higher with batching -- matching §22.5's "individual queries
experience longer QKV projection latency, but the whole group of queries
finish with shorter total latency."

## Build and run

```
make -C part3-advanced-patterns-and-applications/ch22-algorithm-selection-and-problem-decomposition run
```
