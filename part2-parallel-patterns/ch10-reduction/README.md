# Chapter 10: Reduction

Source: *Programming Massively Parallel Processors*, 5th ed., Ch. 10 (pp. 221-250).

| File | Book section | Technique |
|------|-------------|-----------|
| `01_reduction_naive.cu` | §10.3 | Simple kernel (Fig. 10.5): single block, owner position `2*threadIdx.x`, active-thread test `threadIdx.x % stride == 0`, stride doubling 1→blockDim.x |
| `02_reduction_reduced_divergence.cu` | §10.4 | Convergent kernel (Fig. 10.8): owner position `threadIdx.x`, active-thread test `threadIdx.x < stride`, stride halving blockDim.x→1 -- fewer divergent warps |
| `03_reduction_reduced_memory_divergence.cu` | §10.5 | No new kernel in the book -- head-to-head comparison of files 01/02's kernels from the memory-coalescing angle, reproducing the book's own N=256 global-memory-request arithmetic |
| `04_reduction_shared_memory.cu` | §10.6 | Shared-memory kernel (Fig. 10.10): first tree level done directly from global memory outside the loop, remaining levels stay in shared memory, input array left untouched |
| `05_reduction_warp_shuffle.cu` | §10.7 | Warp-shuffle kernel (Fig. 10.14): builds on file 04, switches to `__shfl_down_sync`-based `warp_reduce` (Fig. 10.13) once only one warp remains |
| `06_reduction_two_stage_warp_wide.cu` | §10.8 | Two-stage warp-wide kernel (Fig. 10.16): every warp independently warp-reduces its own segment first (no shared memory), then one warp combines the per-warp partial sums |
| `07_reduction_arbitrary_length.cu` | §10.9 | Multi-block atomic kernel (Fig. 10.18): builds on file 06, partitions input into per-block segments, combines block results with `atomicAdd` |
| `08_reduction_thread_coarsening.cu` | §10.10 | Coarsened kernel (Fig. 10.20): builds on file 07, `COARSE_FACTOR=4` elements per thread loaded and summed in registers before the tree starts |

All eight files perform a **sum reduction** of `float` values -- the chapter's
running example (Fig. 10.3(b), the 8-element sum-reduction tree). Every file
uses the same deterministic synthetic input generator (`generateInput`,
duplicated per file per this repo's self-contained-file convention): a
small linear-congruential PRNG producing uniform `float`s in `[0, 1)`. Every
CPU reference (`reduction_cpu`) implements Fig. 10.1's sequential sum,
accumulated in `double` so the reference itself doesn't add extra rounding
noise on top of the GPU-vs-CPU summation-order difference already inherent
to the comparison (see below).

**On floating-point comparison tolerance.** §10.2 states directly: "strictly
speaking, floating-point additions are not associative due to potentially
different rounding results for different ways of introducing parentheses.
Some applications accept floating-point operation results to be the same if
they are within a tolerable difference from each other." Every kernel here
reorders the addition tree relative to the CPU's strictly-sequential Fig.
10.1 reference, so results are checked with `nearlyEqual(gpu, cpu, 1e-2f)`
(a wider tolerance than this repo's usual `1e-3f` default, chosen because
files 07/08 sum up to 1,024,000 uniform-random values, where the two
summation orders' rounding can diverge by more than one part in a
thousand) rather than exact equality.

## §10.3 Simple kernel -- `01_reduction_naive.cu`

Implements Fig. 10.5 exactly: for an N-element input (N a power of two,
N <= 2048), one block of N/2 threads. Thread `t` owns `input[2*t]`; each
iteration, threads whose `threadIdx.x` is a multiple of the current
`stride` (starting at 1, doubling every iteration) add `input[i+stride]`
into their owned location. The kernel overwrites `input` in place, and
`__syncthreads()` at the end of every iteration is both barrier and memory
fence (§10.3's own closing point).

Measured (`nvcc --resource-usage -arch=sm_75`): 13 registers/thread, 0
bytes shared memory.

## §10.4 Reducing control divergence -- `02_reduction_reduced_divergence.cu`

Implements Fig. 10.8: owner position becomes `threadIdx.x` and stride now
starts at `blockDim.x`, halving to 1, with active threads being the
contiguous prefix `threadIdx.x < stride`. §10.4 derives the concrete
256-element improvement directly: execution-resource
consumption drops from 864 units (Fig. 10.5) to 384 units, raising
utilization efficiency from 255/864 = 0.30 to 255/384 = 0.66 -- "almost
double."

Measured: 10 registers/thread (down from file 01's 13 -- no `%` operator
needed for the active-thread test), 0 bytes shared memory.

## §10.5 Reducing memory access divergence -- `03_reduction_reduced_memory_divergence.cu`

§10.5 introduces **no new kernel figure** -- it re-examines the exact same
two kernels from files 01/02 through the lens of global-memory coalescing.
This file reflects that directly: it re-declares both kernels and runs
them head-to-head (each with its own untimed warm-up launch immediately
before its own timed launch, per this project's timing-fairness
convention, since both are timed in the same process) on identical input.

The naive kernel's owned locations (`2*threadIdx.x`) are never adjacent
within a warp, so "twice the number of global memory transactions as that
for a coalesced access are triggered" even in the first iteration, growing
worse each iteration as `stride` grows. §10.5 derives an exact total for a
256-element input:

```
(N/64*5*2 + N/64 + N/64/2 + ... + 1) * 3 = (4*5*2 + 4+2+1) * 3 = 141
```

(`N/64` = 4 warps launched; the first 5 iterations each have >=2 active
threads per warp, so each active warp issues 2 divergent memory requests;
the remaining iterations have exactly 1 active thread per warp, issuing 1
request each, halving warps every iteration; `*3` = 2 reads + 1 write per
active thread per iteration). This file reproduces that exact arithmetic
in `bookNaiveMemoryRequests256()` and prints it for the N=256 test case --
it is the book's own worked number, not a profiler measurement, and the
comment says so.

For the convergent kernel, owned locations are always adjacent within a
warp, so "the adjacent threads in each warp always access adjacent
locations in the global memory so the accesses are always coalesced" --
§10.5 only asserts this qualitatively (no competing total is derived in
the text), so this file does not invent a number for it; the timing
columns speak to the real effect instead.

## §10.6 Reducing global memory accesses -- `04_reduction_shared_memory.cu`

Implements Fig. 10.10: each thread loads and adds its two elements
directly from global memory once, writes the sum into shared memory, and
all further tree levels read/write shared memory only, with
`__syncthreads()` moved to the top of the loop. §10.6 derives the
resulting DRAM traffic as just `N+1` accesses, `(N/32)+1` requests once
coalesced -- for N=256, 9 requests, "a 4x improvement" over the 36
requests the book states directly for file 02's kernel on the same input.
Unlike files 01-03, `input` is read-only here (§10.6: "useful if the
original values of the array are needed for some other computation").

Measured: 10 registers/thread, 0 bytes *static* shared memory (the
`input_s` array is sized dynamically per launch, `blockDim.x * 4` bytes,
via the kernel's third `<<<>>>` launch argument, so it doesn't show up in
`--resource-usage`'s static count).

## §10.7 Warp-level primitives -- `05_reduction_warp_shuffle.cu`

Implements Fig. 10.12/10.13/10.14: the `warpIdx()`/`laneIdx()` helpers and
the `warp_reduce` device function (`__shfl_down_sync`-based, stride 16→1,
5 = log2(32) steps) are defined verbatim as the book presents them. The
shared-memory loop from file 04 is truncated to stop once `stride` reaches
32 (leaving exactly 32 live partial sums, one warp's worth), after which
only warp 0 continues, loading its 32 values into registers and calling
`warp_reduce` -- replacing what would have been 5 more
`__syncthreads()`+shared-memory rounds with register-only shuffles.

Measured: 12 registers/thread (up from file 04's 10 -- the extra state is
`warp_reduce`'s loop-local `partialSum` and its `__shfl_down_sync` operands).

## §10.8 Two-stage warp-wide reduction -- `06_reduction_two_stage_warp_wide.cu`

Implements Fig. 10.16: rather than starting in shared memory, EVERY warp
immediately warp-reduces its own loaded pair-sum in registers (stage 1, no
shared memory, no `__syncthreads()`), then lane 0 of each warp writes its
warp's total into a 32-slot shared array indexed by `warpIdx()`. One
`__syncthreads()` -- the *only* block-wide barrier in the whole kernel --
separates the stages; then warp 0 loads the (up to 32) per-warp partials
and warp-reduces them again for the final result. §10.8: "The only time we
need to access shared memory and perform barrier synchronization is when
transitioning between the two stages," traded against more control
divergence during stage 1 (no warp gets to drop out early the way file
05's shared-memory phase let entire warps go idle).

Measured: 13 registers/thread, 128 bytes static shared memory
(`WARP_SIZE * sizeof(float)` = `32*4`, matching the fixed 32-slot
per-warp-partial-sum array).

## §10.9 Arbitrary length inputs -- `07_reduction_arbitrary_length.cu`

Implements Fig. 10.18, built directly on file 06: each block is assigned a
`2*blockDim.x`-element segment (`segment = blockIdx.x * 2*blockDim.x`),
runs the same two-stage warp-wide reduction as file 06 on its segment, and
instead of writing its result, accumulates it into the shared output with
`atomicAdd` -- "the thread blocks can make their contributions in any
arbitrary order... the operator for the reduction must be both commutative
and associative," which sum addition is (§10.2, tolerance caveat noted
above). This removes the single-block 2048-element ceiling of files 01-06:
test cases here reduce 20,480 and 1,024,000 elements across 10 and 500
blocks respectively. Following Fig. 10.18 itself, N is required to be an
exact multiple of the per-block segment size (2048 for the 1024-thread
blocks used here) -- handling a remainder is Exercise 5, a reader exercise
out of this project's scope, not chapter content.

Measured: 14 registers/thread, 128 bytes static shared memory (same
32-slot array as file 06).

## §10.10 Thread coarsening -- `08_reduction_thread_coarsening.cu`

Implements Fig. 10.20, built directly on file 07: each block's segment
grows to `COARSE_FACTOR * 2 * blockDim.x` elements, and each thread sums
`COARSE_FACTOR*2` (= 8, for `COARSE_FACTOR=4`) globally-loaded elements
into one register with a `__syncthreads()`-free loop before the same
warp-reduce / one-shared-memory-hop / warp-reduce / `atomicAdd` tail as
file 07. §10.10's own argument: fewer, larger blocks means fewer
redundant per-block reduction-tree overhead episodes (Fig. 10.21: 2
uncoarsened blocks take 8 total steps, 1 block coarsened by 2x doing the
same work takes only 6). This project's directly-derived consequence for
`COARSE_FACTOR=4`: to reduce the same 1,024,000-element input, file 07
launches 500 blocks (and therefore 500 `atomicAdd` contentions on the
single output) while file 08 launches exactly 125 -- a 4x reduction in
block count, matching `COARSE_FACTOR`, confirmed by both files' own
printed block counts below.

Measured: 22 registers/thread (up from file 07's 14 -- the coarsening
loop's induction variable and the extra live global-memory addresses),
128 bytes static shared memory (unchanged).

## Results (RTX 4090, `sm_89`, binaries built for `-arch=sm_75`; this
environment's default CUDA device 0 is the 4090, not the 2070 SUPER)

```
== bin/01_reduction_naive ==
N=128: cpu=55.982853 gpu=55.982857  0.0044 ms  [match]
N=512: cpu=256.274028 gpu=256.274048  0.0051 ms  [match]
N=2048: cpu=1009.515422 gpu=1009.515503  0.0061 ms  [match]
PASS
== bin/02_reduction_reduced_divergence ==
N=128: cpu=55.982853 gpu=55.982857  0.0051 ms  [match]
N=512: cpu=256.274028 gpu=256.274048  0.0041 ms  [match]
N=2048: cpu=1009.515422 gpu=1009.515442  0.0041 ms  [match]
PASS
== bin/03_reduction_reduced_memory_divergence ==
N=256: cpu=125.979841  naive=125.979843 (0.0044 ms) [match]  convergent=125.979843 (0.0041 ms) [match]
  Book's §10.5 worked example (Fig. 10.5, N=256): 141 global memory requests (uncoalesced) -- reproduced from the text's own arithmetic, not measured here.
N=512: cpu=256.274028  naive=256.274048 (0.0041 ms) [match]  convergent=256.274048 (0.0041 ms) [match]
N=2048: cpu=1009.515422  naive=1009.515503 (0.0061 ms) [match]  convergent=1009.515442 (0.0041 ms) [match]
PASS
== bin/04_reduction_shared_memory ==
N=128: cpu=55.982853 gpu=55.982857  0.0050 ms  [match]
N=512: cpu=256.274028 gpu=256.274048  0.0039 ms  [match]
N=2048: cpu=1009.515422 gpu=1009.515442  0.0040 ms  [match]
PASS
== bin/05_reduction_warp_shuffle ==
N=128: cpu=55.982853 gpu=55.982857  0.0049 ms  [match]
N=512: cpu=256.274028 gpu=256.274048  0.0040 ms  [match]
N=2048: cpu=1009.515422 gpu=1009.515442  0.0041 ms  [match]
PASS
== bin/06_reduction_two_stage_warp_wide ==
N=256: cpu=125.979841 gpu=125.979836  0.0051 ms  [match]
N=512: cpu=256.274028 gpu=256.274048  0.0041 ms  [match]
N=2048: cpu=1009.515422 gpu=1009.515442  0.0041 ms  [match]
PASS
== bin/07_reduction_arbitrary_length ==
N=20480 (10 blocks): cpu=10252.471601 gpu=10252.472656  0.0051 ms  [match]
N=1024000 (500 blocks): cpu=512128.677731 gpu=512128.562500  0.0061 ms  [match]
PASS
== bin/08_reduction_thread_coarsening ==
N=81920 (10 blocks): cpu=41036.784843 gpu=41036.781250  0.0052 ms  [match]
N=1024000 (125 blocks): cpu=512128.677731 gpu=512128.750000  0.0051 ms  [match]
PASS
```

At these input sizes every kernel finishes in a few microseconds and is
dominated by fixed per-launch overhead rather than the algorithmic
differences the chapter analyzes (those differences matter at the much
larger N and lower-end hardware the book's own worked examples target);
the register/shared-memory counts measured above and the block-count
reduction in files 07/08 are the quantitative claims this README backs
directly, per this project's convention of not asserting a performance
number without deriving it from the book's text or measuring it on real
hardware.

Every file times its own kernel(s) with a discarded warm-up launch before
the timed region (so PTX->SASS JIT cost isn't folded into the
measurement); file 03 is the one file in this chapter that times two
kernels in-process, and gives each its own independent warm-up immediately
before its own timed launch, per this project's timing-fairness rule.

Build and run all samples in this chapter:

```sh
make run
```
