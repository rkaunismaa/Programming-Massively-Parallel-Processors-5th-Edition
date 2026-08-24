# Chapter 9: Histogram

Source: *Programming Massively Parallel Processors*, 5th ed., Ch. 9 (pp. 201-220).

| File | Book section | Technique |
|------|-------------|-----------|
| `01_histogram_basic_atomics.cu` | §9.2 | Basic kernel (Fig. 9.6): one thread per pixel, direct `atomicAdd`-equivalent (`cuda::atomic_ref` + `fetch_add`) into the public/global histogram, device scope |
| `02_histogram_privatized_shared_mem.cu` | §9.4 | Privatized kernel (Fig. 9.10): each block gets a private histogram in shared memory, block-scope atomics during the pass, device-scope atomics to merge into the public histogram at the end |
| `03_histogram_coarsened.cu` | §9.5 | Coarsened kernel (Fig. 9.14): builds on file 02, adds thread coarsening with **interleaved** partitioning (the strategy the book recommends for GPUs, over contiguous partitioning) so fewer private copies need to be initialized/merged |
| `04_histogram_thread_level_privatization.cu` | §9.6 | Thread-level-privatized kernel (Fig. 9.15): builds on file 03, each thread tracks a single most-recently-seen bin in a register and only commits to shared memory when the run of identical pixels ends |

All four files histogram the **pixel intensities of a synthetic grayscale
image** (`unsigned char`, 256 possible values, one bin per value), matching
the book's own example: "the histogram of pixel intensity values of the
grayscale image of a tree" (Fig. 9.1) — this chapter's histogram is
explicitly an **image** histogram, not a text histogram. Per §9.2's
commentary on the sequential reference (Fig. 9.2), "we assume that each
histogram bin represents a single pixel value," so `NUM_BINS = 256`.

Each file generates a deterministic synthetic image (`generateImage`) built
from runs of identical pixel values (1-24 pixels per run, mean 12.5) whose
overall brightness distribution mirrors Fig. 9.1's own worked example: of
every 64 pixels, 6 are black `[0-63]` (trunk), 12 dark gray `[64-127]`
(leaves), 14 light gray `[128-191]` (grass), 32 white `[192-255]` (sky) —
i.e. roughly 9%/19%/22%/50%. This is the same input generator (duplicated
per file per this repo's self-contained-file convention) across all four
files, so the four kernels can be compared on identical, realistic,
contention-heavy data: heavily skewed toward a few bins (as §9.3 says real
image histograms tend to be, which is exactly what drives up atomic
contention) and containing the identical-value runs that §9.6 says
thread-level privatization specifically exploits ("in pictures of the sky,
there can be large patches of pixels of identical value").

Every kernel's atomic operations use the CUDA C++ `cuda::atomic_ref` API
that the book introduces in §9.2 (`#include <cuda/atomic>`), not the older
free-function `atomicAdd`, per Fig. 9.6's own listing.

## §9.1-9.3 Basic atomics — `01_histogram_basic_atomics.cu`

One thread per input pixel (Fig. 9.6): each thread reads its pixel value
`b` and calls `fetch_add(1, cuda::memory_order_relaxed)` on a
`cuda::atomic_ref<unsigned int, cuda::thread_scope_device>` over `bins[b]`.
Because updates to the same bin must be serialized whether they come from
threads in the same block or different blocks, the reference has **device**
scope. §9.3 explains why this is slow under contention: an atomic
read-modify-write on a given memory location cannot overlap with another
atomic on the same location, so throughput on a single hot bin is bounded
by roughly `1 / (2 x memory latency)` — the book's own worked example
(200-cycle DRAM latency) gives "one atomic operation every 400 cycles ...
2.5 M atomics/second" for a single bin, boosted only by however many bins
the traffic actually spreads across in practice.

Measured (`nvcc --resource-usage -arch=sm_75`): 8 registers/thread, 0 bytes
shared memory — no privatization overhead, but see the timing table below
for the contention cost this pays instead.

## §9.4 Privatization — `02_histogram_privatized_shared_mem.cu`

Implements Fig. 9.10 directly (the book's shared-memory privatized kernel,
which improves on the global-memory-private-copy version of Fig. 9.9 shown
earlier in §9.4). Each block declares a private `bins_s[256]` in shared
memory, zeroes it cooperatively, then runs the same one-thread-per-pixel
pass as file 01 except the atomic target is `bins_s[b]` with
`cuda::thread_scope_block` — contention during this phase is limited to the
threads of one block, and shared-memory latency ("a few cycles" per §9.4)
replaces DRAM/L2 latency. After a `__syncthreads()`, each thread walks a
stride of the 256 private bins, skips zero entries, and atomically commits
nonzero counts into the public `bins` array with `thread_scope_device`
(cross-block contention here is modest: at most one thread per block
touches any given public bin during the merge).

Measured: 8 registers/thread (same as file 01 — no extra per-thread state,
just a different scope/target for the same atomic call), 1024 bytes shared
memory per block (`256 * sizeof(unsigned int)`, exactly the `bins_s[256]`
array, matching §9.4's own arithmetic).

## §9.5 Thread coarsening — `03_histogram_coarsened.cu`

Builds on file 02 (identical shared-memory privatization, init, and commit
phases) and replaces the one-pixel-per-thread pass with a `COARSE_FACTOR=4`
coarsening loop using **interleaved partitioning** (Fig. 9.13/9.14): thread
`threadIdx.x` in block `blockIdx.x` processes pixels
`segment + c*blockDim.x + threadIdx.x` for `c` in `[0, COARSE_FACTOR)`,
where `segment = blockIdx.x * blockDim.x * COARSE_FACTOR`. In each fixed
`c`, consecutive `threadIdx.x` values touch consecutive image addresses, so
every iteration's global load coalesces — unlike contiguous partitioning
(Fig. 9.11/9.12), which the book says "results in a sub-optimal memory
access pattern" on a GPU because each thread's own run of pixels is
non-consecutive across the warp. This reduces the number of blocks (and
thus the number of private-histogram init/merge overheads paid) by roughly
`COARSE_FACTOR`, at the cost of allowing up to `COARSE_FACTOR` more atomics
per bin per block-invocation before the merge — the tradeoff §9.5 identifies
between per-block overhead and intra-block contention.

Measured: 11 registers/thread (up from file 02's 8 — the extra state is the
`count`, `segment`, and `c` loop-control values the coarsening loop needs),
1024 bytes shared memory per block (unchanged from file 02 — `NUM_BINS` and
the private-histogram layout are identical, only the pixel-assignment loop
changed).

## §9.6 Thread-level privatization — `04_histogram_thread_level_privatization.cu`

Builds on file 03's coarsening loop and adds a third privatization level
(Fig. 9.15): rather than an atomic add to `bins_s` on every pixel, each
thread tracks a single `(currVal, accum)` pair in registers — the bin value
currently being accumulated and its run length so far. When the next pixel
matches `currVal`, the thread just does `++accum` (a plain register op, no
atomic at all). When it differs, the thread commits the accumulated
`accum` to `bins_s[currVal]` in **one** atomic add (instead of one atomic
per pixel in that run), then starts tracking the new bin. Because "the
update is always at least one element behind" (§9.6), one more atomic
commits the thread's last tracked run after the coarsening loop exits. This
only pays off when consecutive pixels a thread sees repeat the same value —
true by construction for this chapter's synthetic image (mean run length
12.5, per the shared `generateImage` description above) but not for
uniformly random data, which §9.6 itself flags: "if the similarity between
consecutive pixel values is low, the kernel may execute at slower speed
than the previous kernel."

Measured: 12 registers/thread (up from file 03's 11 — `currVal` and `accum`
are the added thread-private state), 1024 bytes shared memory per block
(unchanged — the block-level privatized histogram itself is identical to
files 02/03; only the per-thread pixel-processing loop changed).

## §9.3, §9.7-9.8 Notes not given a separate file

- **§9.3 atomic throughput analysis** (the 400-cycle / 2.5M-atomics/sec /
  640M-atomics/sec-with-256-bins worked example) is a quantitative
  discussion with no new kernel listing; its conclusions motivate file 02
  and are cited above rather than given a separate file.
- **§9.7 Summary / §9.8 Exercises** are conceptual recap and computed
  problems with no new kernel listing.

## Timing (RTX 2070 SUPER, `sm_75`, 1920x1080 = 2,073,600 pixels)

Every file times its single kernel with a discarded warm-up launch before
the timed region (so PTX->SASS JIT cost isn't folded into the measurement),
matching this repo's standard practice. These are separate binaries with
one kernel each rather than an in-process comparison, so there is no
shared-warm-up-vs-cold-kernel bias between them:

| File | Time | vs. file 01 |
|------|------|------|
| `01_histogram_basic_atomics` | 0.761 ms | 1.0x (baseline) |
| `02_histogram_privatized_shared_mem` | 0.053 ms | ~14x faster |
| `03_histogram_coarsened` | 0.023 ms | ~33x faster |
| `04_histogram_thread_level_privatization` | 0.022 ms | ~35x faster |

This ordering matches the book's narrative: privatization gives the single
largest jump (moving almost all contention off the device-scope public
histogram and onto much-lower-latency shared memory), coarsening adds a
further improvement (fewer blocks means less init/merge overhead), and
thread-level privatization gives a smaller additional improvement here
because the synthetic image's mean run length (12.5) means most of the
per-pixel atomics were already being coalesced into shared memory cheaply
by file 03 — the technique's benefit scales with how much run-length
similarity the real input data has (§9.6).

Build and run all samples in this chapter:

```sh
make run
```
