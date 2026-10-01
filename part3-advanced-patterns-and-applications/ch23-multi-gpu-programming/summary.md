# Chapter 23 — Multi-GPU Programming

Section-by-section summary of the book content. Page numbers are printed
book pages (Chapter 23 runs pp. 541–576).

All prior chapters targeted one host with one device. This chapter
covers scaling to **clusters** with multiple hosts and multiple GPUs,
via three programming models: **MPI** (Message Passing Interface),
**NCCL** (NVIDIA Collective Communications Library), and **NVSHMEM**
(NVIDIA's OpenSHMEM implementation) — covering only the key concepts
needed to scale an application across GPUs/nodes, not full API
coverage (readers are pointed to each library's own tutorials for
that). The chapter focuses on domain partitioning, point-to-point
communication, collective communication, and overlapping computation
with communication, using one running example throughout.

## 23.1 Stencil as a running example (p. 542)

The running example is the **Jacobi iterative method** on a 2D
structured grid (a simplified, 2D version of Chapter 8's stencil
pattern): each grid point's new value is the average of its 4
neighbors from the previous iteration, repeated until the residual
(difference between old and new values, summarized as an **L2 norm**)
drops below a tolerance. Fig. 23.1's single-GPU kernel extends
Chapter 8's stencil kernel with two changes: it computes a 2D (not 3D)
stencil, and it additionally computes each point's residual and
atomically accumulates a block-reduced L2-norm contribution into a
global `l2norm` (reusing Chapter 10's reduction pattern) — further
optimizations (shared-memory tiling, thread coarsening, CUB's
`BlockReduce`) are explicitly out of this chapter's scope.

**Domain partitioning**: with a row-major 2D array, it's natural to
partition the grid along the **y-dimension**, giving each GPU a
contiguous set of rows (Fig. 23.2) — contiguous in memory, matching
the row-major layout. Grid points at a partition's edge need the
**old** values of the adjacent partition's edge rows to compute their
**new** values — these needed-but-foreign rows are **halo** points
(Fig. 23.3); after computing, a GPU's own halo-adjacent rows become the
*new* halo data its neighbors need, so before the next iteration, each
GPU must **exchange** its newly-computed boundary rows with its
neighbors — called **halo exchange**. 1D (row-based) partitioning keeps
halo rows memory-contiguous (simplifying the exchange) but can need
transmitting an entire wide row per exchange if the domain is very
wide; a 2D tiled partitioning reduces total communicated data
(better surface-to-volume ratio) at the cost of needing 4 neighbors
instead of 2 and non-contiguous halo data — the chapter sticks with
simple 1D partitioning throughout. Every iteration thus has two
phases: **computation** (new grid-point values) and **communication**
(halo exchange, plus an L2-norm reduction across all partitions).

## 23.2 Multi-GPU stencil with MPI (p. 545)

Parallel programming models split into **shared memory** (all threads/
processes share one address space — e.g. CUDA threads within a device)
and **distributed memory** (separate address spaces, requiring
explicit message-passing to share data). **MPI** is the dominant
distributed-memory interface for compute clusters: processes
communicate by sending/receiving messages, addressed via logical **MPI
ranks** (analogous to phone numbers) rather than needing interconnect
details. MPI is SPMD, like CUDA.

**Setup/teardown** (Figs. 23.4–23.6): `MPI_Init()` initializes the
runtime; `MPI_Comm_rank(comm, &rank)` returns the calling process's
unique rank within a **communicator** (`MPI_Comm` — a named group of
processes, `MPI_COMM_WORLD` by default meaning "all processes");
`MPI_Comm_size(comm, &numRanks)` returns the communicator's total rank
count (queried at runtime rather than hard-coded, since it's set by the
user at launch via `mpirun`/`mpiexec`); `MPI_Finalize()` tears down MPI
resources. Each rank allocates its own partition's GPU memory
(`cudaMalloc`), initializes it, and sets boundary conditions, before
entering the main iterative loop.

**Main loop** (Fig. 23.7): resets the L2 norm, launches the Jacobi
kernel for the full local partition, waits (`cudaDeviceSynchronize`),
performs halo exchange, reduces the L2 norm across ranks, and
double-buffers (`std::swap`) input/output for the next iteration.
**Halo exchange** is **point-to-point communication** — exchanging
messages between two specific processes. The two most basic
primitives are `MPI_Send()`/`MPI_Recv()` (Figs. 23.8–23.9) —
**two-sided**: both a sender and a matching receiver must participate.
Since each rank must *simultaneously* send to one neighbor and receive
from the other, the chapter uses the fused `MPI_Sendrecv()` (Fig.
23.10) instead of separate send/receive calls (non-blocking
`MPI_Isend`/`MPI_Irecv`/`MPI_Wait` are named as an alternative for more
complex patterns, not used here). A **periodic boundary condition**
(wrap-around) treats the topmost rank's "top neighbor" as the
bottommost rank and vice versa, avoiding special-casing the true grid
edges. Two `MPI_Sendrecv()` calls per iteration handle the top and
bottom exchange directions respectively.

A key detail: the `output`/`input` pointers passed to `MPI_Sendrecv`
are **GPU device pointers** — only possible with **CUDA-aware MPI**
(an MPI implementation designed to accept CUDA device pointers
directly, eliminating manual host-staging copies and their added
latency; most major implementations — MPICH, OpenMPI, MVAPICH2 — are
CUDA-aware).

**Collective communication** (contrasted with point-to-point) involves
**all** ranks in a communicator at once (barrier, broadcast, reduce,
gather, scatter, etc.) and is typically far better optimized than
hand-rolled sequences of point-to-point calls. The L2-norm reduction
uses `MPI_Allreduce()` (Fig. 23.11): reduces one buffer per rank (here,
one float — that rank's local L2-norm-squared) with a given operation
(`MPI_SUM`) and gives **every** rank a copy of the combined result. The
L2-norm-squared is copied device→host *before* calling `MPI_Allreduce`
specifically because the *result* is needed on the host anyway (for the
convergence check), so copying first is more efficient than reducing
on-device and copying after.

## 23.3 Overlapping computation and communication (p. 553)

The §23.2 implementation strictly alternates compute-only and
communicate-only phases, under-utilizing one resource (GPU compute or
the interconnect) at any given moment. The fix: split each process's
work into **two stages** (Fig. 23.12) — Stage 1 computes just the
**boundary** rows first (needed as halo data by neighbors next
iteration); Stage 2 then computes the **internal** rows *while
simultaneously* exchanging the just-computed boundary rows with
neighbors — if internal computation takes longer than the halo
exchange, the exchange's latency is fully hidden (Fig. 23.13 compares
timelines with/without this overlap). Achieving this needs splitting
the Jacobi kernel call into **three** separate kernel launches (top
boundary, bottom boundary, internal), run **concurrently** via three
**CUDA streams** (`topStream`, `bottomStream`, `internalStream`) — a
stream is an ordered sequence of GPU operations; operations in the
*same* stream execute in order, operations in *different* streams may
execute concurrently with no ordering guarantee. Kernel launches are
**asynchronous** regardless of stream: the host call returns as soon as
the operation is *inserted* into its stream, not when it actually
starts or finishes executing.

Because the order the three concurrent kernels actually execute in
isn't guaranteed, the kernels computing boundary rows are given
**higher stream priority** than the internal-rows kernel (via
`cudaStreamCreateWithPriority`, using the range from
`cudaDeviceGetStreamPriorityRange`), ensuring boundary data is ready to
communicate as early as possible rather than risking the internal
kernel monopolizing GPU resources first.

Splitting the kernel into three streams raises two synchronization
challenges, both solved with CUDA **events** (a marker for a specific
point in a stream, which other streams or the host can wait on): (1)
the halo exchange for a boundary row must wait until *that stream's*
kernel finishes — solved with `cudaStreamSynchronize()` on `topStream`/
`bottomStream` before each respective `MPI_Sendrecv`; (2) all three
kernels update the *same* global `l2norm`, so its reset-to-zero
(`cudaMemsetAsync`, placed in `internalStream`) must complete before
*any* of the three kernels start, and all three kernels must complete
before the L2 norm is copied/reduced — solved via a `resetL2` event
recorded after the memset and waited on (`cudaStreamWaitEvent`) in
`topStream`/`bottomStream`, plus `computeTop`/`computeBottom` events
recorded after their respective kernels and waited on in
`internalStream` before its `cudaMemcpyAsync` of the L2-norm value
(Fig. 23.14, fully illustrated in the timeline of Fig. 23.15).

An important caveat raised directly from this timeline: **host-side
stream synchronization introduces real latency** — e.g. `MPI_Exchange
Bottom` cannot even *begin* until the blocking `cudaStreamSynchronize`
for the bottom stream returns on the host, which itself must wait for
`MPI_Exchange Top` to finish first (since both are issued sequentially
by the single host thread) — genuinely limiting how much overlap is
achievable. Reducing exactly this kind of host-synchronization overhead
is a central design goal of NCCL and NVSHMEM (§23.4–23.5).

A final implementation detail: `cudaMemcpyAsync` requires the host
memory buffer involved (`l2norm_h`) to be **pinned** (page-locked)
memory, allocated via `cudaMallocHost` rather than ordinary `malloc` —
explained via a sidebar on virtual memory and DMA: `cudaMemcpy`'s DMA
hardware operates on physical addresses, but the OS can relocate
(page out) ordinary virtual memory at any time; CUDA's default
`cudaMemcpy()` avoids this risk by first staging through an internal
pinned buffer, but doing so forces the whole copy to be
**synchronous** — `cudaMemcpyAsync()` instead requires the *caller* to
supply already-pinned memory, eliminating that internal staging step
and the synchronous behavior that came with it.

## 23.4 Multi-GPU stencil with NCCL (p. 563)

§23.3's remaining inefficiency is the host's synchronous role as
"synchronizing intermediary" between kernel calls and MPI calls — e.g.
blocking `cudaStreamSynchronize()` calls before each `MPI_Sendrecv`.
**NCCL** addresses this directly: its communication primitives
(point-to-point send/receive, and collectives — reduce, broadcast,
scatter, gather) run **on the GPU**, not the host CPU, and can
therefore be placed **inside CUDA streams** just like kernel calls —
freeing the host from needing to synchronize between the two. NCCL is
also **topology-aware**, automatically optimizing for the available
interconnect (PCIe, NVLink, InfiniBand Verbs, sockets) without the
programmer needing to hand-tune for each. NCCL is not a full MPI
replacement, but is commonly used **alongside** MPI.

**Setup/teardown** parallels MPI: `ncclGetUniqueId()` (called by one
rank only), `ncclCommInitRank()` (called by every rank, given that
shared unique ID plus the rank's own number and the total rank count,
to build an `ncclComm_t` communicator), and `ncclCommDestroy()` (Fig.
23.16). Since the shared unique ID must reach every rank before
`ncclCommInitRank` is called, Fig. 23.17 uses MPI itself (`MPI_Bcast`
+ `MPI_Barrier`) to distribute it — MPI and NCCL working together.

For halo exchange, the relevant NCCL primitives are `ncclSend()`/
`ncclRecv()` (Fig. 23.18) — closely resembling `MPI_Send`/`MPI_Recv`,
but **asynchronous** and taking a CUDA stream parameter. Since
`MPI_Sendrecv`'s fused send+receive has no direct NCCL equivalent, NCCL
instead provides **group calls**: `ncclGroupStart()`/`ncclGroupEnd()`
(Fig. 23.19) bracket a set of primitives (here, one `ncclSend` + one
`ncclRecv`) so they execute together as one fused operation, without
host synchronization in between. Fig. 23.20 integrates this into the
main loop: each `MPI_Sendrecv` call is replaced by a
`ncclGroupStart()`/`ncclSend()`/`ncclRecv()`/`ncclGroupEnd()`
sequence, passed the relevant stream directly instead of a prior
`cudaStreamSynchronize()`.

An important subtlety: `ncclGroupEnd()` returning only guarantees the
primitives have been **inserted** into their streams — not that the
actual communication has **completed**. So (unlike the blocking
`MPI_Sendrecv`) explicit events (`exchangeTop`/`exchangeBottom`,
recorded right after the group calls) are still needed, waited on by
`internalStream` before the next iteration's L2-norm reset proceeds
(Fig. 23.21's timeline shows the halo exchange has moved entirely into
the `topStream`/`bottomStream` timelines, with the host now mostly
freed from communication/synchronization duties except waiting on
`internalStream` before the final `MPI_Allreduce`).

## 23.5 Multi-GPU stencil with NVSHMEM (p. 568)

Both MPI and NCCL use **two-sided** communication: a send has no
effect until a *matching* receive call executes on the other side, so
both processes must actively participate and wait for each other to
reach their respective calls — adding inherent latency. **One-sided**
communication is an alternative: a process can directly `put` data
into, or `get` data from, another process's address space **without**
that other process executing any matching call — reducing latency, and
(unlike matched two-sided calls, impractical to coordinate at the scale
of hundreds of thousands of device threads) making it practical for
individual **GPU threads** to directly initiate communication.
**NVSHMEM** is NVIDIA's implementation of the OpenSHMEM one-sided
communication standard for GPU clusters, supporting puts/gets
initiated from host threads **or individual device threads**. NVSHMEM
calls a process/rank a **processing element (PE)**.

Since a one-sided operation's initiator must itself specify the
*target* process's address (no matching receive call to supply it),
NVSHMEM uses a **symmetric heap**: every PE allocates a portion of its
address space identically (same relative offset) for cross-PE
communication, so the initiator can compute a target's address just by
knowing its *own* corresponding local address — the runtime handles
translating offset→actual target address. `nvshmem_malloc()`/
`nvshmem_free()` manage symmetric allocations; `nvshmem_float_g()` /
`nvshmem_float_p()` are device-callable get/put functions for single
float values (Fig. 23.22) — **all PEs must call the allocation
routines together**, and the input/output grid partition layout is
assumed identical (symmetric) across all PEs.

Fig. 23.23's kernel: each thread computes its stencil update as
before, but a boundary-row thread (`y==1` or `y==ny-2`) additionally
calls `nvshmem_float_p()` to directly **push** its newly-computed value
into the correct halo slot of the appropriate **neighboring PE's**
symmetric array — halo exchange happens **inline inside the
compute kernel itself**, with no separate communication call or
kernel needed at all. Setup (Fig. 23.24) again layers on top of MPI:
`nvshmemx_init_attr()` initializes NVSHMEM using an existing MPI
communicator (via `nvshmemx_init_attr_t`'s `mpi_comm` field and the
`NVSHMEMX_INIT_WITH_MPI_COMM` flag) so PE/rank numbering aligns with
MPI ranks; `nvshmem_finalize()` releases NVSHMEM resources at the end.

Fig. 23.25's main loop needs only **one** kernel launch per iteration
(not three, since there's no separate boundary/internal split — the
halo `put` is embedded directly in the single kernel), into a single
stream. Because `nvshmem_float_p` is asynchronous and doesn't itself
guarantee delivery, an explicit **barrier**,
`nvshmemx_barrier_all_on_stream(stream)`, is needed after the
device-to-host L2-norm copy/reduce step, to guarantee **all** PEs'
put operations have actually landed before the next iteration begins —
placed on the same stream as the kernel so it, too, needs no host-side
polling/synchronization to enforce ordering.

NVSHMEM's two key advantages over MPI/NCCL: (1) computation and
communication overlap **automatically** — since the put is literally
embedded in the compute kernel, there's no need to split kernels or
orchestrate concurrency via separate streams/events; (2) it enables
**fusing multiple logically-separate operations into one kernel**
(not very visible with this chapter's single-operation stencil example,
but valuable for more complex pipelines with communication interleaved
between several computational steps), which can reduce kernel-launch
overhead and keep data resident in registers/shared memory across
operations that would otherwise need separate kernels. One caveat
flagged for further study: the chapter's **fine-grained**, per-thread
put calls work well over fast, low-latency interconnects (NVLink), but
coarser-grained puts (e.g. `nvshmemx_float_put_block`, where a whole
thread block collaborates on one larger put) are more efficient over
slower network interconnects.

## 23.6 Summary (p. 574)

The chapter covered multi-GPU domain partitioning and halo exchange,
using Jacobi stencil iteration as the running example, across three
programming models of increasing communication sophistication. **MPI
alone** gives a straightforward but non-overlapped implementation.
Splitting computation into boundary/internal stages and using **CUDA
streams** enables overlap, but the host must still synchronize
explicitly between kernel calls and MPI's two-sided, host-blocking
communication. **NCCL** communication primitives run on the GPU and fit
inside CUDA streams directly, freeing the host from that
synchronization burden, and are pre-optimized for the system's actual
interconnect topology. **NVSHMEM**'s one-sided communication removes
the need for a matching receive call entirely and lets individual
device threads initiate puts/gets — enabling automatic computation/
communication overlap and even fusing multiple operations into a
single kernel. The comparison deliberately highlighted NCCL's
*host-initiated, two-sided* style against NVSHMEM's *device-initiated,
one-sided* style, though both libraries are actively evolving toward
overlapping feature sets (e.g. NCCL gaining symmetric-memory and
device-initiated support; NVSHMEM already supporting host-initiated
communication too) — host-initiated communication suits coarse-grained,
computation-independent data; device-initiated suits fine-grained data
whose communication is determined as part of the computation itself.
Despite multi-GPU programming's apparent unfamiliarity, its core
concepts (SPMD, ranks, barriers) all have direct CUDA counterparts —
reinforcing the book's belief that mastering one parallel programming
model well (CUDA) makes picking up others straightforward.

## 23.7 Exercises (p. 576)

Three problems: a multi-part exercise computing, for a 64×512-point
5-point stencil partitioned across 16 MPI ranks — the output grid
points, halo grid points, Stage-1 boundary grid points, Stage-2
internal grid points, and bytes sent per process (per Fig. 23.12) (Q1);
determining the per-element size (1/2/4/8 bytes) implied by a given
`MPI_Send` call's parameters and total byte count (Q2); and identifying
which of three statements about MPI blocking/message-size behavior is
true (Q3). Not implemented in this repo's samples (end-of-chapter
exercises are out of scope — see the repo root README).
