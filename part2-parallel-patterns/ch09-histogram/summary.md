# Chapter 9 — Histogram (Atomic Operations and Privatization)

Section-by-section summary of the book content. Page numbers are printed
book pages (Chapter 9 runs pp. 201–220).

Unlike every pattern so far, histogram cannot use the **owner-computes
rule** — the output location a thread writes to is *data-dependent*
(which bin an input element falls into), so multiple threads may need
to update the *same* output element. This introduces the chapter's two
central themes: **race conditions**/**atomic operations** (how to make
concurrent updates to the same location safe) and **privatization**
(how to make them fast despite that safety requirement).

## 9.1 Background (p. 202)

A **histogram** counts, for each interval ("bin") of a value range, how
many data elements fall in that interval, typically plotted as bars
(Fig. 9.1 — a grayscale tree image's pixel intensities split into 4
bins of 64 values each). Histograms are a common, useful data-analysis
summary: their shape reveals things like exposure bias in an image, or
anomalous purchase patterns for fraud detection. Applications include
computer-vision feature extraction, speech recognition, recommendation
systems, and scientific data analysis. A sequential histogram (Fig.
9.2) is a simple `O(N)` loop that reads each element and does
`++bins[b]`; it's typically memory-bound on a CPU, but efficient
because the sequential `image` access pattern makes full use of cache
lines and the small `bins` array fits comfortably in L1 cache.

## 9.2 Atomic operations and a basic histogram kernel (p. 203)

The obvious parallelization — one thread per input element, each doing
`++bins[b]` for its own element (Fig. 9.3) — creates **output
interference**: multiple threads may be assigned pixels with the same
value and so need to update the *same* bin concurrently. An increment
is a **read-modify-write** sequence (read old value, add 1, write new
value) — the same hazard behind real-world bugs like two airline
customers both being told they hold seat 9C after simultaneously
reading/modifying/writing the same seat-map entry. When two threads'
read-modify-write sequences don't fully complete before the other
starts (Figs. 9.4–9.5), the result depends on their relative timing —
a **race condition** — and some interleavings silently lose an update
(final count off by one from the correct value).

An **atomic operation** makes a read-modify-write sequence on a memory
location indivisible: no other read-modify-write to that *same*
location can overlap with it, enforced by hardware locking. Atomics do
*not* impose any particular order between threads (either thread may
run first) — only that whichever runs second cannot start until the
first fully completes, which serializes (but does not reorder) updates
to one location. CUDA exposes this via the C++ `<cuda/atomic>` library:
`cuda::atomic_ref<T, cuda::thread_scope>` constructs an atomic
reference to an existing object (as opposed to `cuda::atomic`, which
declares an object atomic from creation) — the scope parameter
(`cuda::thread_scope_block`, `..._device`, etc.) controls which threads
the atomicity is guaranteed to be visible to. Fig. 9.6's
`histogram_kernel`: one thread per pixel (bounds-checked), builds
`cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
bins_ref(bins[b])`, then calls `bins_ref.fetch_add(1,
cuda::memory_order_relaxed)` — `fetch_add`'s first argument is the
value to add (1, matching the referenced object's type); the second,
`memory_order_relaxed`, tells the compiler/hardware no additional
reordering restriction is needed beyond what the thread's own
instruction-level data dependency on `b` already enforces (a stronger
ordering becomes necessary in Chapter 11's use case). `fetch_add`
returns the pre-update value, which this kernel discards (it only
cares about the final count) — but is used by later patterns (Chapters
12, 18).

## 9.3 Latency and throughput of atomic operations (p. 209)

Serializing updates to a contended location is intrinsically costly:
each atomic operation's duration is roughly a DRAM load latency plus a
DRAM store latency (hundreds of cycles total, Fig. 9.7), and because
updates to the *same* location must fully serialize, the achievable
throughput on one heavily-contended location is just
`1/(load_latency+store_latency)` — e.g. 200+200 cycles at 1 GHz gives
only **2.5 M atomics/sec** on one location, far below the GPU's raw
memory throughput (32 G elements/sec in the chapter's worked numeric
example), since ordinary (non-atomic, non-conflicting) accesses *can*
have many in flight simultaneously while atomics to one location
cannot. Spreading atomics across the 256 histogram bins helps (up to
256× in the uniform case, 640 M atomics/sec) — but real image data is
rarely uniformly distributed (Fig. 9.1's own example is biased toward
bright bins), so contention — and the throughput penalty — concentrates
on whichever bins happen to be popular.

Modern GPUs mitigate this by supporting atomics with device scope
directly in the **last-level (L2) cache**, shared across all SMs: a
contended variable found in L2 is updated right there (tens of cycles)
rather than requiring a full DRAM round trip (hundreds of cycles) — and
since heavily-atomic-contended variables are, by definition, heavily
accessed, they tend to stay resident in L2 across many threads'
updates, giving at least an order-of-magnitude throughput improvement
over naive DRAM-latency-bound atomics. This cache-assist is necessary
but, as the arithmetic above shows, often still insufficient — the
chapter's subsequent sections attack the problem from the software
side instead.

## 9.4 Privatization (p. 211)

**Privatization** is a general technique for parallel programs with
heavy output interference: replicate a contended output data structure
into several **private copies**, assign disjoint subsets of threads to
update different private copies (so contention only occurs *within* a
subset), then **merge** the private copies into the public one once all
updates are done — merging is straightforward here because addition is
commutative and associative, so private copies can be summed into the
public copy in any order. The tradeoff is that reduced contention must
outweigh the added merge cost; in practice privatization is applied per
*group* of threads (e.g. per block) rather than per individual thread,
since per-thread privatization of a large histogram would itself be
prohibitively expensive (taken up directly in §9.6).

Fig. 9.9's kernel gives each **thread block** its own private histogram
in a `bins_pool` region of global memory (block `i`'s private bins at
offset `i*NUM_BINS`); the first phase is identical to Fig. 9.6 except
atomics now target `bins_priv` with `thread_scope_block` (a narrower
scope than `thread_scope_device`, since only same-block threads
contend on it — this can let hardware achieve lower latency than a
full device-scope atomic). After a `__syncthreads()`, each thread
commits a subset of the block's private bins to the public `bins` array
via a *device-scope* atomic add (needed here since multiple blocks
commit concurrently) — but since each location is touched by only one
thread per block during this phase, contention during the commit phase
is modest. Because private copies are themselves in global memory, they
are likely to be cached, reducing their effective access latency.

If the number of bins is small enough, the private copy can instead be
declared directly in **shared memory** (Fig. 9.10) — collapsing the
DRAM-latency atomic into a shared-memory-latency one (a few cycles vs.
hundreds), directly multiplying achievable throughput. On GPUs
supporting thread block clusters, privatization can also be done at the
**cluster** level, using cluster-wide barriers and the cluster's
distributed shared memory — allowing a larger effective private
histogram than a single block's shared memory could hold.

## 9.5 Thread coarsening (p. 214)

Privatization's cost is the per-block overhead of initializing and
later committing a private copy — paid once per block, so launching
*more* blocks pays it more often. This overhead is worth paying when
blocks genuinely run concurrently in parallel, but wasted if the
hardware ends up serializing excess blocks anyway (not enough SM
capacity to run them all at once). **Thread coarsening** reduces the
block count (hence the number of private-copy initialize/commit
cycles) by having each thread process multiple input elements — explored
via two partitioning strategies:

- **Contiguous partitioning** (Figs. 9.11–9.12): each block's segment
  size is `COARSE_FACTOR * blockDim.x`, and within that segment each
  thread handles one *contiguous* run of `COARSE_FACTOR` elements
  (`i = segment + threadIdx.x*COARSE_FACTOR + c`). This is the natural,
  good strategy on a CPU (few concurrent threads, so a thread's whole
  contiguous run stays undisturbed in that thread's cache line across
  its iterations) — but performs poorly on a GPU, where many threads in
  an SM compete for the same cache, so a single thread's sequential
  run isn't reliably still cached by the time it circles back.
- **Interleaved partitioning** (Figs. 9.13–9.14): the only code change
  is the index formula (`i = segment + c*blockDim.x + threadIdx.x`) —
  now, on each coarsening iteration, the *whole warp* of threads
  accesses consecutive elements together (thread 0 → `segment+0`,
  thread 1 → `segment+1`, etc. on iteration `c=0`; thread 0 →
  `segment+blockDim.x`, etc. on iteration `c=1`), which **coalesces**
  exactly like Chapter 6's coalescing patterns — motivated directly by
  that chapter's observation that GPU cache behavior under heavy thread
  concurrency differs fundamentally from a CPU's.

## 9.6 Thread-level privatization (p. 217)

Coarsening opens up a *third* level of privatization: since a coarsened
thread now processes several input elements sequentially itself, it can
keep a **thread-private** running bin value and only commit it with an
atomic operation when necessary — skipping the atomic entirely for runs
of *consecutive, same-valued* elements the thread processes itself (a
situation common in data with large localized patches of identical
value, e.g. a sky region of near-uniform pixel intensity). This is
infeasible to do for *every* bin per thread (too much private storage
for a large histogram), so Fig. 9.15 privatizes only **one bin at a
time**: the most recently seen one. Each thread loads its first pixel
and sets a thread-private counter `bin_r = 1` for that bin; on each
subsequent pixel, if it matches the currently-tracked bin, the thread
just increments `bin_r` locally (no atomic at all); if it differs, the
thread first atomically commits the *old* `bin_r` to the block-private
`bins_s` (shared memory), then resets `bin_r = 1` for the new pixel's
bin. Because the committed count is always "one pixel behind," a final
atomic commit for the last-tracked bin is needed after the loop ends.

This technique is a net win only when consecutive elements a thread
processes are often identical — for low-repetition data, the extra
bookkeeping (comparison, extra branch) can make the kernel *slower*
than without thread-level privatization, while for highly repetitive
data it can be substantially faster. The added `if` does introduce
control divergence risk, but in the two extremes (no similarity at all,
or very high similarity) threads in a warp tend to agree on which
branch to take, so divergence is naturally limited to the mixed middle
ground — and even there, it's typically outweighed by the throughput
gained from avoiding atomics altogether.

## 9.7 Summary (p. 219)

Histogram is an important real-world computation and a natural vehicle
for two foundational parallel-programming concepts: **race conditions**
in read-modify-write sequences, and **atomic operations** as the
correctness mechanism for resolving them. Because atomic throughput on
one location is roughly the inverse of twice the memory access latency,
heavy contention can make a naively-parallelized histogram
surprisingly slow — motivating **privatization** (replicating the
contended structure so that, among other benefits, atomics can target
low-latency shared memory instead of global memory) and **thread
coarsening** (reducing the number of private copies that must be
initialized and merged, via contiguous or interleaved partitioning).
**Thread-level privatization** pushes privatization one level further,
to a per-thread single-bin cache, trading added bookkeeping for a
potentially large reduction in atomic operations on data with high
value-locality.

## 9.8 Exercises (p. 220)

Numeric problems computing atomic-operation throughput under various
assumptions: a single DRAM-latency figure (Q1); an L2-cache-hit-rate
model (Q2); the resulting FLOP/s ceiling for a kernel performing a
fixed number of FLOPs per atomic op (Q3); the same with privatization
into shared memory, including its own added overhead (Q4); and a
multi-part question computing the exact number of global-memory atomic
operations for a specific 524,288-element/128-bin histogram problem
under three kernel variants — no privatization (Fig. 9.6), block
privatization without coarsening (Fig. 9.10), and block privatization
with 4× coarsening (Fig. 9.14) (Q5). Not implemented in this repo's
samples (end-of-chapter exercises are out of scope — see the repo root
README).
