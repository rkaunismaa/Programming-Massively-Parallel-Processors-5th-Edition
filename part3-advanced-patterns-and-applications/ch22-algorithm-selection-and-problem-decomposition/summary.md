# Chapter 22 — Algorithm Selection, Problem Decomposition, and Problem Formulation

Section-by-section summary of the book content. Page numbers are printed
book pages (Chapter 22 runs pp. 529–540).

This chapter is retrospective and conceptual rather than code-bearing:
it generalizes parallel programming into an abstract **thinking
process** — selecting algorithms, decomposing a domain problem into
coordinated parallel work units, and formulating problems to expose
parallelism — synthesizing patterns scattered across every earlier
chapter rather than introducing new kernels. Good algorithm selection
and decomposition let a programmer balance parallelism, work
efficiency, and resource consumption; strong parallel-thinking skills
let a programmer **collaborate with domain scientists** to restructure
a domain problem (identifying which parts are inherently serial vs.
parallelizable), rather than simply implementing problems handed down
unchanged.

## 22.1 Algorithm selection (p. 530)

An algorithm must be **definite** (every step precisely stated),
**effectively computable** (every step computer-executable), and
**finite** (guaranteed to terminate). Given a problem, multiple valid
algorithms typically exist, differing in algorithmic complexity
(steps/work required), degree of exposed parallelism, generality, and
numerical accuracy/stability — rarely is one algorithm best on every
axis, so selecting the best compromise for the target hardware is a
core parallel-programming skill. The chapter revisits three earlier
examples as worked illustrations:

- **Prefix sum** (Chapter 11): **Brent-Kung** has lower algorithmic
  complexity (more work-efficient) but **Kogge-Stone** exposes more
  parallelism (fewer sequential steps) — a classic complexity-vs.-
  parallelism tradeoff, often mitigated by hybrid approaches (combining
  two parallel algorithms, or a parallel algorithm with a lower-
  complexity sequential one via thread coarsening).
- **Sorting** (Chapter 14): odd-even sort has the highest complexity
  but is trivially parallelizable; radix sort achieves lower complexity
  (non-comparison-based) and is highly parallel-amenable, but isn't
  generally applicable (needs specific key types); merge sort is more
  general (works with any well-defined comparison operator) at higher
  complexity than radix sort — a complexity-vs.-generality tradeoff.
- **Electrostatic potential map** (Chapter 21): DCS and cutoff
  summation both expose ample parallelism but trade algorithmic
  complexity against accuracy — cutoff summation sacrifices a small
  amount of accuracy for dramatically better scaling (linear rather
  than quadratic in system volume). This tradeoff isn't unique to
  parallel programming (it exists sequentially too), but it introduces
  an **additional parallelization-specific challenge**: a naive
  sequential cutoff algorithm is **atom-centric** (each atom scans
  nearby grid points), which doesn't parallelize well due to scatter
  memory access; the solution needed a **grid-centric** reformulation
  (cutoff **binning**, Chapter 21) — all blocks iterate their own
  neighborhood bins, with threads making individual per-atom cutoff
  decisions. Binning's own subtlety (bins of uneven atom count) was
  handled via a bounded bin size plus host-processed **overflow list**.

## 22.2 Problem decomposition (p. 532)

Once an algorithm is chosen, the problem must be decomposed into
sub-problems solvable concurrently — conceptually simple but often
practically hard; the key is identifying the right **units of work**
per thread (or thread group) to fully exploit the problem's inherent
parallelism. Two common strategies (Fig. 22.1):

- **Output-centric decomposition**: each thread (or group) produces
  one or more **output** elements, reading whatever input elements its
  sub-problem needs (possibly all inputs, e.g. grid-centric DCS; or a
  subset, e.g. matrix multiplication, grid-centric cutoff binning).
- **Input-centric decomposition**: each thread (or group) processes
  one or more **input** elements, applying their contribution to
  whichever output element(s) they affect (one output, e.g. histogram;
  a subset, e.g. hypothetically atom-centric cutoff summation; or all
  outputs, e.g. atom-centric DCS).

Some problems (e.g. odd-even sort, Chapter 14) have such simple,
direct input↔output relations that the two strategies coincide. Where
they differ, they typically exhibit opposite **memory access
patterns**: output-centric decomposition usually **gathers** (Fig.
22.1a — multiple input values combined into a thread's private
output), which CUDA devices handle well (results accumulate in private
registers, inputs can be shared via cache/shared memory, conserving
global bandwidth); input-centric decomposition usually **scatters**
(Fig. 22.1b — one thread's result distributed across multiple output
locations), which is usually undesirable, since concurrently-written
shared output locations require **atomic operations** — substantially
slower than output-centric's private register accesses. Beyond
gather-vs-scatter, other considerations also matter: exposed
parallelism, ease of identifying which inputs map to which outputs,
and resulting load balance.

A detailed worked walkthrough revisits decomposition choices across
the whole book:

- **Image processing (Ch. 3), matmul (Ch. 3/5/15), convolution
  (Ch. 7), stencil (Ch. 8)** — output-centric: avoids atomics (gather),
  and no other consideration favors input-centric here (ample exposed
  parallelism either way, straightforward input↔output mapping,
  uniform per-output work with no load imbalance risk).
- **Histogram (Ch. 9)** — **input**-centric (needs scatter/atomics):
  the output-centric alternative would face three problems —
  drastically reduced parallelism (far fewer bins than input values),
  no cheap way for a thread to know which inputs map to its bin without
  scanning everything (not work-efficient), and severe load imbalance
  (wildly varying inputs-per-bin counts) — together outweighing the
  atomics cost of the input-centric choice.
- **Reduction (Ch. 10) / scan (Ch. 11)** — output-centric per
  iteration (each thread gathers two elements from the current
  iteration's input into one output); the input-centric alternative
  would scatter contributions from multiple inputs into the same
  output element, again needing atomics.
- **Unstable/stable filter (Ch. 12)** — input-centric: each thread
  owns an input element and decides where (if at all) it lands in the
  output; an efficient output-centric alternative would need each
  output-owning thread to inspect a large number of other elements,
  causing redundant work. Notably, the *stable* filter's overall
  input-centric decomposition still internally **uses** an
  output-centric sub-operation (scan) — illustrating that a single
  complex computation can combine multiple decomposition strategies at
  different levels.
- **Merge (Ch. 13)** — output-centric, even though it avoids no
  atomics (each output receives from exactly one input, so an
  input-centric merge wouldn't need atomics either): the real reason
  is **load balance under thread coarsening** — an input-centric merge
  partitioning one input array across threads risks wildly uneven
  per-thread workloads (and resulting control divergence), which
  output-centric avoids.
- **Sorting (Ch. 14)** — decomposition choice varies by algorithm:
  odd-even sort's input-centric and output-centric views coincide;
  merge sort is output-centric (built on merge); radix sort is
  input-centric (each thread owns an input element, computes its new
  output position — a generalization of the stable-filter pattern).
- **Wavefront/dynamic programming (Ch. 16)** — output-centric: each
  thread gathers multiple previously-computed values for its assigned
  table cell; an input-centric alternative would scatter one computed
  value to all succeeding dependent cells, again needing atomics
  (multiple inputs can contribute to the same succeeding cell).
- **Sparse matrix computation (Ch. 17)** — CSR/ELL/JDS SpMV kernels
  are output-centric (gather a row's non-zeros); COO SpMV is
  input-centric (scatter, atomics) but gains more exposed parallelism
  and better load balance in exchange; CSC SpMV is also input-centric
  (its format makes an output-centric row-scan prohibitively
  expensive) — the best choice is genuinely dataset-dependent, and the
  **hybrid ELL-COO** format is a concrete example of combining both
  strategies within one solution.
- **Graph traversal / BFS (Ch. 18)** — vertex-centric push and
  edge-centric BFS are input-centric (threads assigned to vertices/
  edges scatter level updates to neighbors); vertex-centric pull is
  output-centric (each thread gathers into only the vertex it owns).
  Since BFS level updates are idempotent and atomic-free regardless,
  the gather/scatter distinction itself isn't the deciding factor here
  — exposed parallelism and load balance (dataset-dependent) are what
  actually decide which is better (see Chapter 18 for detail).
- **Electrostatic potential map (Ch. 21)** — output-centric
  (grid-centric) is favored specifically for its gather-pattern
  avoidance of atomics; both decompositions expose sufficient
  parallelism, so the atomics consideration dominates here, with
  binning resolving the remaining input-to-output mapping difficulty
  for the cutoff algorithm.

## 22.3 Application level considerations — Amdahl's law (p. 537)

Real applications combine multiple modules with uneven workloads (Fig.
22.2's molecular-dynamics example: vibrational/rotational forces vs.
non-bonded forces, typically far more compute-intensive). A programmer
must judge whether each module's work volume justifies GPU
implementation — small modules may stay on the host while a kernel
handles only the dominant module, with the host combining both
modules' results afterward.

**Amdahl's Law** governs the resulting application-level speedup:
worked example — non-bonded force calculation is 95% of sequential
runtime and gets 100× GPU speedup, while the remaining 5% stays on the
host with no speedup: if host and device execution are **sequential**
(not overlapped), overall speedup is `1/(5% + 95%/100) = 1/5.95% ≈
17×`; if host and device work can be **overlapped** (the small
host-side portion fully hidden behind the GPU's execution), overall
speedup rises to `1/5% = 20×`. Either way, the *unaccelerated* 5%
caps total speedup far below the kernel's own 100× — illustrating a
major challenge in scaling large applications: the accumulated runtime
of many small, individually-not-worth-parallelizing activities can
become the actual bottleneck end users experience. This often
motivates **task-level parallelism**: running several such smaller
activities concurrently (e.g. on a multi-core host, or as multiple
small concurrent kernels using CUDA streams — Chapter 23) even if each
alone wouldn't achieve great standalone GPU speedup.

## 22.4 Problem formulation (p. 538)

**Problem formulation** — how a domain problem is mathematically/
computationally framed in the first place — is "arguably the most
consequential aspect of parallel application development." Simply
mapping a domain problem onto generic numerical methods using only the
skills taught so far often yields high computational complexity and/or
limited parallelism; genuinely effective solutions often require
**rethinking the underlying numerical method itself** to relax
constraints that block efficient parallel implementation. The
chapter's standing example: cutoff binning (Chapter 21) required
*domain* expertise (physics-based justification for trading accuracy)
*combined with* problem-decomposition/optimization skill (grid-centric
decomposition plus a binning data structure) — a genuinely
interdisciplinary effort whose payoff can be transformative (e.g.
enabling high-fidelity simulation of biochemical systems previously
considered infeasible).

## 22.5 Batching: latency vs. throughput (p. 539)

All of the book's optimizations so far targeted **latency**: making
one fixed input run faster (**latency optimization**). But at least
one more motivation exists for optimizing parallel programs:
**throughput** — processing many separate problem instances with
better overall turnaround, even at the cost of each individual
instance's own latency.

Worked motivating scenario: a financial firm needs its 10-hour
portfolio-risk analysis cut to 4 hours — ordinary latency optimization
(e.g. Chapter 20's flash attention eliminating redundant global memory
round trips) targets exactly this. A second, less intuitive scenario:
the firm wants to run **many instances** of the analysis (different
parameters) and the current program can't finish them all within the
window — here, **batching** multiple instances together can improve
*total* throughput even if it makes any single instance take *longer*.
Chapter 20's `QKV` projection batching is the running example: batching
converts many small vector-matrix multiplications into one larger
matrix multiplication; the larger multiplication takes **more time**
than any one original vector-matrix op (so **individual latency
increases**), but far **less time** than the **sum** of all the
original operations (since reusing the shared weight matrix across the
batch eliminates redundant global memory accesses) — raising
**arithmetic intensity** and therefore raising the batch's overall
**throughput** (`total queries / batch latency` > `total queries / sum
of individual latencies`). This latency-vs.-throughput tradeoff
parallels Chapter 1's own CPU-vs.-GPU design philosophy contrast
(CPU: optimized for low per-element latency; GPU: optimized for high
aggregate throughput).

## 22.6 Summary (p. 540)

The chapter surveyed parallel programming's core *thinking* steps:
**problem formulation**, **algorithm selection**, **problem
decomposition**, and (as covered throughout the rest of the book)
**optimization**. Algorithms trade off complexity, parallelism,
generality, and accuracy; decomposition choices for a given algorithm
affect inter-thread interference, exposed parallelism, load balance,
and other performance factors; and **batching** can trade increased
individual latency for increased aggregate throughput across a group
of problem instances. The book deliberately teaches these ideas
bottom-up — grounded first in concrete CUDA implementations — on the
premise that hands-on experience with a specific programming model
builds the maturity needed to generalize parallel-thinking skills
beyond GPUs specifically. No code-bearing exercises accompany this
chapter (it has no §22.7 Exercises section — the chapter closes with
its Summary and References).
