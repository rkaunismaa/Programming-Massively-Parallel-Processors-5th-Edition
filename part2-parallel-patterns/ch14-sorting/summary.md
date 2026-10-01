# Chapter 14 — Sorting

Section-by-section summary of the book content. Page numbers are printed
book pages (Chapter 14 runs pp. 329–348).

## 14.1 Background (p. 330)

A sorting algorithm arranges list elements into nondecreasing or
nonincreasing order while remaining a permutation of the input (no
elements lost, duplicated, or altered). Elements can be sorted directly
by value, or by a separate **key field** attached to each element's
value (e.g. sorting (age, income) tuples by income). A sort is
**stable** if elements with equal keys keep their original relative
order — essential for cascaded multi-key sorts, where a stable sort on
the secondary key followed by a stable sort on the primary key
correctly produces an overall primary-then-secondary ordering.
**Comparison-based** sorts (bubble sort, merge sort, etc.) are
provably bounded below by `O(N·logN)`; some **non-comparison-based**
algorithms (e.g. radix sort) can beat that bound, at the cost of not
generalizing to arbitrary key types. The chapter covers two
comparison-based algorithms (odd-even sort, merge sort) and one
non-comparison-based algorithm (radix sort), chosen to illustrate a
range of parallelization and optimization techniques rather than to be
exhaustive.

## 14.2 Parallel odd-even sort (p. 331)

**Bubble sort** repeatedly compares and swaps adjacent out-of-order
pairs in sequence until the list is sorted; since the order pairs are
visited in doesn't actually matter for correctness, a parallel variant
can sort many pairs **simultaneously** instead. The catch: two threads
sorting *overlapping* pairs (e.g. indices (0,1) and (1,2)) would race on
the shared element. **Odd-even (transposition) sort** avoids this by
alternating, each iteration, between sorting all **even pairs**
(indices (0,1), (2,3), ...) and all **odd pairs** (indices (1,2),
(3,4), ...) — within either phase, all pairs are disjoint, so all
comparisons/swaps in that phase are safely independent (Fig. 14.1).
Iterations keep alternating even/odd until a full pass produces no
swaps, meaning the list is sorted.

Fig. 14.2's kernel is called once per iteration, with half as many
threads as list elements; an `isOddStep` parameter selects whether
thread `t` sorts pair `(2t, 2t+1)` (even step) or `(2t+1, 2t+2)` (odd
step); each thread compares and conditionally swaps its pair, setting a
global `hasChanged` flag on any swap so the host knows another
iteration is needed. Multiple threads writing `1` to the *same* flag is
technically a race condition, but a **benign** one since every writer
writes the identical value — called **idempotence**. The book notes
this is nonetheless undefined behavior under the strict C++ memory
model even though it works in practice, so a conservative programmer
should use an atomic write instead.

Odd-even sort's complexity is poor: worst case (largest element at the
list's start) needs `O(N)` iterations to migrate it to the end, and each
iteration does `O(N)` comparisons even with unlimited execution
resources — giving `O(N)` time complexity (with unlimited parallelism)
and `O(N²)` work complexity, motivating the search for better
algorithms.

## 14.3 Parallel merge sort (p. 333)

Merge sort divides the input into segments, sorts each segment
independently, then repeatedly merges pairs of sorted segments until one
fully-sorted segment remains (Fig. 14.3) — parallelism can be extracted
both **across** independent merge operations at a given stage and
**within** each merge operation itself (using Chapter 13's parallel
merge). There's an inherent tradeoff across stages: early stages have
many small, independent merges (more cross-merge parallelism, less
within-merge parallelism); late stages have few large merges (the
reverse) — e.g. an 8-block grid might assign 2 blocks to each of 4
first-stage merges, but 4 blocks to each of 2 second-stage merges, to
keep the GPU's resources matched to the available parallelism at each
stage.

With `O(logN)` merge stages, each costing `O(logN)` time and `O(N·logN)`
work (per Chapter 13) if fully parallelized, the overall parallel merge
sort is `O(log²N)` time and `O(N·log²N)` work — the work bound is
**worse** than sequential merge sort's `O(N·logN)`, specifically because
the parallel *merge* operation itself (via the co-rank binary search)
does more work than sequential merging. (Cole's algorithm, cited but not
covered, achieves `O(logN)` time and `O(N·logN)` work, closing this
gap.) A full implementation is left as an exercise.

## 14.4 Radix sort (p. 334)

To beat `O(N·logN)`, a non-comparison-based algorithm is needed. **Radix
sort** distributes keys into buckets based on a **radix value** (a
digit or fixed-size group of bits), repeating for each digit position
until all digits are covered; each iteration is **stable**, which is
essential since later iterations rely on earlier iterations' relative
ordering being preserved within a bucket. For binary keys, a
power-of-two radix is natural (a 1-bit radix: 2 buckets per iteration,
one per bit value). Fig. 14.4 works a complete 4-bit, 1-bit-radix sort
by hand across 4 iterations: iteration 1 buckets by least-significant
bit, iteration 2 by the next bit (with iteration-1's order preserved
within each new bucket), and so on up through the most significant bit
— after all 4 iterations, the list is fully sorted. One radix sort
iteration resembles Chapter 12's **stable filter**, generalized: instead
of keeping only elements meeting a condition, *every* element is kept,
but partitioned into a "condition true" sub-list and a "condition
false" sub-list, each internally order-preserving — sometimes called
the **stable partition** pattern. This kinship means radix sort's
parallelization/optimization techniques closely mirror stable filter's.

## 14.5 Parallel radix sort (p. 336)

Since each iteration depends on the *complete* result of the previous
one, iterations run sequentially; the chapter focuses entirely on
parallelizing a **single** iteration (host code loops over iterations,
launching the kernel once per iteration). One thread per input key
(Fig. 14.5): each thread must compute its key's **destination index**
in the output list. For a key mapping to the **zero bucket**, its
destination equals the number of *zero-bucket* keys before it, which
equals (its own index) minus (the number of *one-bucket* keys before
it) — `key_index - #ones_before`. For a key mapping to the **one
bucket**, its destination equals (total zero-bucket keys) plus (number
of one-bucket keys before it) = `input_size - #ones_total +
#ones_before`. In both cases, the only non-trivial quantity needed is
`#ones_before` at every position — exactly an **exclusive scan**
(Chapter 11) over a binary "is this key's radix bit 1?" array (Fig.
14.6 walks a full worked example).

Fig. 14.7's `radix_sort_iter` kernel: each thread extracts its key's
relevant bit via shift-and-mask (`(key >> iter) & 1`), stores it to a
`bits` array; a grid-wide exclusive scan (Chapter 11's single-kernel
scan) over `bits` gives `#ones_before` at every position (plus the
scan's final total, `#ones_total`, as a byproduct); each thread then
computes its destination with the two formulas above and writes its key
there.

## 14.6 Optimizing for memory coalescing (p. 339)

The approach in §14.5 writes to the output list in an **uncoalesced**
pattern: consecutive threads scatter their writes between the zero and
one buckets essentially at random relative to each other (worse for
larger radix values, §14.7), so a warp's writes can span many distinct
cache lines. Following Chapter 6's "stage irregular accesses through
shared memory" strategy: each thread **block** first performs a
**local** radix sort of just its own keys in shared memory (using a
*block-local* exclusive scan, not a grid-wide one), then writes its
local buckets out to global memory in one coalesced chunk per bucket
(Fig. 14.8) — since consecutive threads within a block's local
zero-bucket (or local one-bucket) write to *consecutive* global
addresses, this write pattern largely coalesces.

The remaining challenge: each block must know **where** in the *global*
output its local buckets begin. A block's local zero-bucket must start
right after all preceding blocks' local zero-buckets; a block's local
one-bucket must start after *all* blocks' local zero-buckets plus all
preceding blocks' local one-buckets. This is found by building a small
table of each block's local bucket sizes (row-major: all blocks' local
zero-bucket sizes, then all blocks' local one-bucket sizes) and running
an **exclusive scan** over that linearized table (Fig. 14.9) — giving
every block's global starting offset for both of its local buckets in
one scan operation.

## 14.7 Choice of radix value (p. 342)

A larger radix (more bits processed per iteration) needs **fewer total
iterations** to fully sort N-bit keys — e.g. a 2-bit radix sorts 4-bit
keys in 2 iterations instead of 4 (Fig. 14.10), using 4 buckets instead
of 2 each iteration. The same block-local-sort-then-coalesced-write
strategy applies directly (Fig. 14.11), just generalized to `2^r`
buckets for an `r`-bit radix — an `r`-bit local sort is itself performed
as `r` chained 1-bit radix iterations internally (each with its own
local exclusive scan, no cross-block coordination needed for these
*local* sub-iterations), and the cross-block bucket-position table
(§14.6) grows from 2 rows to `2^r` rows (Fig. 14.12).

Fewer iterations means fewer kernel launches, global memory round
trips, and grid-wide exclusive scans — but larger radix values have two
costs: (1) **more, smaller buckets per block** means less data per
coalesced write chunk, so **opportunities for coalescing shrink** as
radix grows; (2) the cross-block exclusive-scan table grows with
`2^r · numBlocks`, so **scan overhead grows** with radix too. Choosing
`r` is thus a genuine three-way tradeoff (iteration count vs.
coalescing quality vs. scan overhead), not a free "bigger is always
better" knob. Implementing a working multi-bit-radix kernel is left as
an exercise.

## 14.8 Thread coarsening to improve coalescing (p. 344)

Having many thread blocks means small local buckets per block (less
coalescing opportunity when writing out) — a cost worth paying only if
those blocks genuinely execute in parallel; if the hardware would
serialize them anyway, the cost is pure waste. **Thread coarsening**
(each thread handling multiple input keys instead of one) directly
addresses this: fewer, larger blocks process more keys each, producing
larger local buckets with more coalescing opportunity when written out
(Fig. 14.13, compared directly against the uncoarsened Fig. 14.11 —
visibly more consecutive-thread/consecutive-address writes). Coarsening
also shrinks the cross-block bucket-position table (fewer blocks =
fewer rows), directly reducing the overhead of the global exclusive
scan from §14.6/14.7. Implementing this is left as an exercise.

## 14.9 Other parallel sort methods (p. 345)

Surveys algorithms beyond the chapter's three, briefly:

- **Sorting networks** (e.g. odd-even transposition sort, §14.2) use a
  **fixed** comparison pattern independent of data values, making them
  trivially parallelizable. **Batcher's bitonic sort** and **odd-even
  merge sort** are the best-known examples, needing only
  `O(N·log²N)` comparisons — asymptotically worse than `O(N·logN)`
  methods like merge sort, but often the fastest choice in practice for
  small sequences due to their simplicity.
- Non-network comparison-based sorts split into two camps: ones that
  focus most work on **merging** sorted tiles (merge sort, this
  chapter's §14.3) and ones that focus most work on **partitioning**
  the unsorted sequence up front, after which combining partitions is
  nearly free (**sample sort**: pick `p-1` sample keys, sort them, use
  them to partition the input into `p` value-ordered buckets — a
  `p`-way generalization of quicksort's 2-way partition — then sort
  each bucket independently and concatenate). Sample sort (with
  over-sampling to balance bucket sizes with high probability) is often
  the best choice for extremely large sequences, including data
  distributed across multiple GPUs.
- Radix sort variants split by digit-processing order: this chapter's
  approach is **LSD (least-significant-digit)** radix sort, working
  from the least to most significant digit. **MSD (most-significant-
  digit)** radix sort instead partitions by the most significant digit
  first, then recurses independently into each resulting bucket using
  the next digit — like sample sort, MSD radix sort is often preferred
  for very large sequences, since (unlike LSD, which needs a global
  reshuffle every iteration) each step operates on progressively more
  localized regions of data.

## 14.10 Summary (p. 346)

The chapter covered two comparison-based algorithms — odd-even sort
(simple, alternating disjoint-pair comparisons, but `O(N²)` work) and
merge sort (building on Chapter 13's parallel merge, extracting
parallelism both across and within merge operations) — and one
non-comparison-based algorithm, radix sort, whose `O(N·logN)`-beating
complexity comes from iteratively bucketing keys digit-by-digit via
stable partitioning. Radix sort's central implementation challenge is
achieving coalesced memory access for its bucket writes; this chapter's
answer is a local-sort-in-shared-memory-then-coalesced-global-write
strategy, further tuned by the choice of radix size (iteration count vs.
coalescing vs. scan overhead) and by thread coarsening (fewer, larger
blocks for better coalescing and less scan overhead). As with earlier
chapters' patterns, production code should generally prefer a
library implementation (Thrust, CUB) over a hand-rolled sort.

## 14.11 Exercises (p. 347)

Four implementation exercises, each extending the single-bit-radix
kernel of Fig. 14.7: add shared memory to improve coalescing (Q1, i.e.
§14.6's optimization); generalize to a multi-bit radix (Q2, i.e.
§14.7's extension); apply thread coarsening to improve coalescing
further (Q3, i.e. §14.8's optimization); and implement a complete
parallel merge sort using Chapter 13's parallel merge (Q4, i.e. §14.3's
deferred implementation). Not implemented in this repo's samples
(end-of-chapter exercises are out of scope — see the repo root README)
— though the repo's own `06_radix_sort_2bit.cu` sample implements
§14.7's 2-bit-radix technique, which the book presents as worked
explanation (Figs. 14.10–14.12) rather than purely as a numbered
exercise.
