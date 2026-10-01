# Chapter 4 — Compute Architecture and Scheduling

Section-by-section summary of the book content. Page numbers are printed
book pages (Chapter 4 runs pp. 67–92).

## 4.1 Architecture of a modern GPU (p. 68)

A CUDA-capable GPU is an array of highly threaded **Streaming
Multiprocessors (SMs)**. Each SM contains several **streaming
processors** (sometimes called "CUDA cores") that share control logic
and on-chip memory. Starting with Hopper, SMs are grouped into **GPU
Processing Clusters (GPCs)** — the H100 has 132 SMs × 128 streaming
processors = 16,896 streaming processors total, arranged into 8 GPCs.
Off-chip device memory (**global memory** / DRAM — GDDR historically,
HBM on newer architectures) is reached through multiple memory
controllers for aggregate bandwidth.

## 4.2 Thread block scheduling (p. 69)

When a kernel launches, the runtime assigns blocks to SMs **as a whole,
block-by-block** — a block is never split across SMs, and all its
threads land on the same SM at the same time. Multiple blocks can share
one SM if resources allow (the number is limited and is the subject of
§4.7). Since the grid usually has far more blocks than can run at once,
the runtime keeps a work-list and assigns new blocks to SMs as earlier
ones finish. This same-SM-same-time guarantee is what makes two things
possible: block-wide barrier synchronization (§4.3) and fast data
exchange through shared memory (Chapter 5). Hopper adds an optional
**thread block cluster** level above blocks: blocks in a cluster are
co-scheduled onto the same GPC and can synchronize and share
*distributed shared memory* with each other.

## 4.3 Synchronization and transparent scalability (p. 70)

`__syncthreads()` is a block-wide barrier: a thread that calls it waits
until every thread in the block has *arrived*, then all are *released*
together, guaranteeing a whole phase of a block's work finishes before
any thread starts the next phase (Fig. 4.3). Cooperative Groups extends
this to cluster-wide and grid-wide barriers, though grid-wide
synchronization is far more heavyweight and restricted (see Chapter 18).

**Strict rule:** if `__syncthreads()` is inside an `if`, either *all*
threads in the block take that branch or *none* do; for `if`/`else`,
each branch needs its own barrier call — but two different call sites
are two **different** barriers even though every thread hits one of
them. Fig. 4.4's `incorrect_barrier_example` shows exactly this trap:
splitting threads by even/odd `threadIdx.x` into branches that each
call `__syncthreads()` looks fine (every thread hits "a" barrier) but
is undefined behavior, since the hardware never guarantees the two
groups wait for each other — it can deadlock or silently corrupt
results.

Because barriers are scoped to one block, blocks never wait on each
other, so the runtime is free to run them in any order and in however
many fit on the hardware at once (Fig. 4.5). This is **transparent
scalability**: the same compiled kernel runs correctly — just slower or
faster — on a cheap GPU with few SMs or a high-end GPU with many,
without any code change. A group of blocks running simultaneously is a
**wave**; total waves = total blocks ÷ blocks-that-fit-at-once. A grid
whose block count doesn't divide evenly leaves a partial final wave
that underutilizes the GPU (the **tail effect**) — worth sizing grids
to avoid.

## 4.4 Warps and SIMD hardware (p. 74)

Once a block is assigned to an SM, hardware further divides it into
32-thread **warps** — the unit of thread scheduling. `threadIdx` values
within a warp are consecutive (warp *n* = threads `32n` … `32(n+1)-1`);
a block whose size isn't a multiple of 32 pads its last warp with
inactive threads. Multidimensional blocks are linearized row-major
(x fastest, then y, then z) before partitioning (Fig. 4.7).

Each SM is organized into **processing blocks**, each with its own
shared instruction fetch/dispatch unit; a warp's threads all execute
the *same instruction* at the same time, differing only in the data
(register contents) they operate on — **Single Instruction, Multiple
Data (SIMD)**. CUDA exposes this as **SIMT** (Single Instruction,
Multiple Thread): programmers write ordinary per-thread scalar code,
and the hardware transparently groups it into warps, unlike classic CPU
SIMD where the programmer/compiler must explicitly manage the SIMD
unit. Sharing one control/fetch unit across many execution units is
what lets GPUs spend most of their die area on arithmetic throughput
rather than control logic.

## 4.5 Control divergence (p. 79)

SIMD execution is efficient when all threads in a warp follow the same
control-flow path. When threads in a warp disagree (e.g. an `if`/`else`
where some take one branch, some the other), the hardware takes
**multiple passes** — one per distinct path — masking off the threads
not on that path during each pass (Fig. 4.9). This is **control
divergence**: it preserves per-thread correctness but costs extra
passes and wastes execution slots on the inactive threads each pass.
Divergence also occurs in loops with a thread-dependent trip count
(Fig. 4.10) — a warp keeps re-executing the loop body, with threads
that finished early sitting inactive, until the last thread's iteration
count is satisfied.

A control construct diverges only if its decision depends on values
that differ within a warp (e.g. `threadIdx` or data-dependent values);
`if (threadIdx.x > 2)` only diverges in the one warp straddling index
2 — other warps are uniformly true or uniformly false. A very common
source is **boundary-condition checks** (`if (i < n)`) when the data
size isn't a multiple of block size — but the book shows this cost
shrinks fast as problems get large: for a 1003-element vector addition
with 64-thread blocks, only 1 of 32 warps diverges (~3% impact, less if
data is larger); for a 200×150 2D image with 16×16 blocks, only 80 of
1040 warps diverge (<8%, trending toward <2% for realistic image
sizes). From Volta onward, **independent thread scheduling** means
diverged passes may interleave rather than run strictly
sequentially, and threads are *not* guaranteed to automatically
reconverge after diverging — correctness across divergent paths now
requires an explicit warp-level barrier such as `__syncwarp()`.

## 4.6 Warp scheduling and latency tolerance (p. 82)

An SM typically has far more warps assigned to it than it can execute
in a given instant — deliberately. When a warp's next instruction
depends on a long-latency operation (e.g. a global memory access) that
hasn't completed, the SM simply doesn't select it; instead it picks
another resident warp that's ready to go. This **fine-grained
multithreading** is how GPUs achieve **latency tolerance** ("latency
hiding"): with enough resident warps, there's almost always another one
ready to execute while others wait. Because all warps' execution state
(registers, etc.) stays resident in the SM's register file the whole
time, switching between warps costs **zero overhead** — unlike a CPU
context switch, which must save/restore state to memory. This is why
GPUs devote relatively little die area to caches and branch prediction
compared to CPUs, favoring more arithmetic units and memory channels
instead.

## 4.7 Resource partitioning and occupancy (p. 85)

**Occupancy** = (warps actually assigned to an SM) ÷ (max warps the SM
supports). SM resources — registers, shared memory, thread slots, and
*block* slots — are **dynamically partitioned** across whatever blocks
are resident, which is flexible (many small blocks or few large ones)
but can produce non-obvious occupancy limits:

- **Block-slot limit:** a Hopper H100 SM supports up to 2048 threads
  but only 32 block slots. A block size of 32 threads would need 64
  blocks to fill 2048 threads, but only 32 blocks fit — capping
  occupancy at 1024/2048 = 50%, even though thread capacity alone
  wasn't exhausted.
- **Indivisibility:** a block size of 768 fits only 2 blocks
  (1536 threads) per SM — neither the thread limit (2048) nor the
  block limit (32) is hit, but 512 thread slots sit unused (75%
  occupancy).
- **Register pressure:** H100 has 65,536 registers/SM. Full occupancy
  at 2048 threads requires ≤32 registers/thread; a kernel using 64
  registers/thread caps out at 1024 threads (50% occupancy)
  regardless of block size. A worked example shows a **performance
  cliff**: going from 31 to 33 registers/thread at a 512-thread block
  size drops the SM from 4 resident blocks to 3 — occupancy falls from
  100% to 75% from a change of just 2 registers per thread.

NVIDIA's **Occupancy Calculator** (in Nsight Compute) and the
`cudaOccupancyMaxActiveBlocksPerMultiprocessor()` API let you compute
real occupancy for a given kernel/launch configuration rather than by
hand.

## 4.8 Querying device properties (p. 87)

Because applications run across a wide range of GPUs, CUDA C++ exposes
runtime queries so host code can discover (rather than hard-code) what
a given device supports: `cudaGetDeviceCount()` returns how many
CUDA-capable devices are present (a modern PC often has more than one,
including a weak integrated GPU), and `cudaGetDeviceProperties()` fills
a `cudaDeviceProp` struct per device. Fields relevant to this chapter:
`maxThreadsPerBlock`, `multiProcessorCount` (SM count),
`clockRate`, `maxThreadsDim[0..2]` / `maxGridSize[0..2]` (per-dimension
limits), `regsPerBlock` (registers available per SM, despite the
name), and `warpSize`. `cudaOccupancyMaxPotentialClusterSize()` plays
the analogous role for thread block clusters. The resource amount per
SM is tied to a device's **compute capability** (e.g. A100 = 8.0, H100
= 9.0) — generally, higher compute capability means more resources per
SM.

## 4.9 Summary (p. 89)

A CUDA GPU is an array of SMs, each built from processing blocks of
streaming processors sharing control logic and memory. Blocks are
assigned to SMs as whole units, in arbitrary order, giving transparent
scalability — with the constraint that blocks in different thread
blocks must not rely on synchronizing with each other. Once on an SM, a
block is divided into 32-thread warps executed SIMD-style; divergent
threads within a warp cost extra passes. SMs oversubscribe themselves
with more resident threads than they can run at once, so they can swap
to a ready warp whenever another is stalled on a long-latency
operation — occupancy (resident threads ÷ max supported) measures how
well-stocked an SM is to hide such latency. Each device's resource
limits (blocks, threads, registers, etc.) are fixed per-SM quantities
that can each become the limiting factor for a given kernel; CUDA C++
provides occupancy calculators/APIs plus runtime device-property
queries so code can adapt to whatever GPU it runs on.

## 4.10 Exercises (p. 90)

Nine problems applying the chapter's concepts by hand: computing warps
per block/grid and counting divergent warps/SIMD efficiency for a given
kernel (Q1); divergence from vector-length boundary checks (Q2–3); time
spent waiting at a barrier given per-thread timings (Q4); why skipping
`__syncthreads()` for a single-warp block is risky even though it may
"work" (Q5); and several occupancy-limit calculations given
block/thread/register budgets (Q6–9). Not implemented in this repo's
samples (end-of-chapter exercises are out of scope — see the repo root
README).
