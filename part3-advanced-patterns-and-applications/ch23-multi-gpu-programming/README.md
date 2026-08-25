# Chapter 23: Multi-GPU programming

Source: *Programming Massively Parallel Processors*, 5th ed., Ch. 23 (pp. 541-576).

| File | Book section | Technique |
|------|-------------|-----------|
| `01_stencil_singlegpu_baseline.cu` | §23.1, Fig. 23.1 | Single-GPU 2D Jacobi iterative-method stencil with block-reduced/atomic L2-norm convergence check -- the chapter's running example, before any domain decomposition |
| `02_stencil_multigpu_mpi.cu` | §23.2, Fig. 23.5-23.11 | Multi-GPU domain decomposition (1D, along y) with MPI: `MPI_Sendrecv` halo exchange, `MPI_Allreduce` L2-norm reduction |
| `03_stencil_multigpu_mpi_overlap.cu` | §23.3, Fig. 23.12-23.15 | Same MPI decomposition, but the Jacobi kernel is split into top-boundary/bottom-boundary/internal launches on three CUDA streams (with priorities and events) so halo exchange overlaps with internal computation |
| `04_stencil_multigpu_nccl.cu` | §23.4, Fig. 23.16-23.21 | Same three-stream overlap structure, but halo exchange uses NCCL (`ncclSend`/`ncclRecv` inside `ncclGroupStart`/`ncclGroupEnd`, placed directly in the CUDA streams) instead of host-blocking `MPI_Sendrecv` |
| `05_stencil_multigpu_nvshmem.cu` | §23.5, Fig. 23.22-23.25 | Halo exchange via NVSHMEM one-sided `nvshmem_float_p` put operations issued directly from inside the Jacobi kernel by the threads computing the boundary rows -- a single kernel/stream per iteration, no manual stream-splitting needed |

**`02`–`05` require MPI / NCCL / NVSHMEM respectively, none of which have
system-wide dev packages installed on this machine; they are written to
match the book's listings but were not compiled or run here.** Only
`01_stencil_singlegpu_baseline.cu` is built and run by `make run` (see the
Makefile). As a syntax/API-shape sanity check during development, `02`-`05`
were each compiled (not linked, not run) against small hand-written stub
headers reproducing the real MPI/NCCL/NVSHMEM function signatures from
memory; all four compiled cleanly with no errors. That check catches typos
and signature mismatches but cannot verify runtime correctness against the
real libraries, since the stub headers are not the real libraries.

## Build and run

```
make -C part3-advanced-patterns-and-applications/ch23-multi-gpu-programming run
```

Only `01_stencil_singlegpu_baseline.cu` is built and executed; `make run`
prints a note after it reminding that `02`-`05` are not built.

## §23.1 -- `01_stencil_singlegpu_baseline.cu`

Fig. 23.1's kernel, as described in the book: each thread computes the x/y
coordinates of its grid point, skips it if the point is on the array's
edge (`x=0`, `x=nx-1`, `y=0`, `y=ny-1` -- "boundary points or halo points,
as we will see later"), otherwise sets its new value to the average of its
4 neighbors' old values, computes the residual (new - old), and
contributes `residual^2` to a block-wide shared-memory reduction that is
atomically added into a grid-wide L2-norm-squared accumulator. At this
point in the chapter there is no domain decomposition at all -- one GPU
owns the whole grid, so the array's edges are genuine, fixed Dirichlet
boundary points, not halo rows belonging to another GPU.

The kernel is written with a generalized `numRows` span parameter instead
of hard-coding `ny`, because §23.3's Fig. 23.14 later reuses this *exact
same* kernel three times per iteration on sub-spans of a rank's local
slab (top boundary row / bottom boundary row / internal rows) -- see
`03_stencil_multigpu_mpi_overlap.cu`'s comments. In this file it is always
called with the full grid (`numRows = ny`), which is exactly Fig. 23.1.

**Test problem** (the book only describes a generic "modeled system"
stencil, giving no concrete numeric example): a 130x130 grid (128x128
interior) modeling 2D steady-state heat conduction with Dirichlet boundary
conditions -- the top edge held at 1.0, the other three edges held at 0.0,
interior initialized to 0.0. This is a standard, deterministic Jacobi
relaxation test problem, chosen because it lets an independently written
CPU reference (double precision, identical iteration structure, run for
the same fixed number of iterations as the GPU) be compared point-by-point
against the GPU result with `nearlyEqual`.

**Measured result (this machine):**

```
Grid: 130x130 (interior 128x128)
GPU: 2000 iterations, final L2 norm = 5.678845e-03
CPU: 2000 iterations, final L2 norm = 5.678847e-03
Max |GPU - CPU| = 1.530310e-07
GPU time: 15.867 ms (2000 iterations)
PASS
```

Both loops run the fixed cap of 2000 iterations (the L2 norm has not yet
dropped below the `1e-6` tolerance at that point -- Jacobi relaxation on a
128x128 grid converges slowly, needing tens of thousands of iterations for
full convergence, which is not the point of this sample). What matters for
correctness is that two *independently implemented* Jacobi loops (GPU
float32 vs. CPU float64), run for the identical iteration count from
identical initial/boundary conditions, agree to `1.5e-7` -- well inside the
default `nearlyEqual` epsilon of `1e-3`.

`compute-sanitizer --tool memcheck` on the built binary reports
**0 errors**.

## §23.2 -- `02_stencil_multigpu_mpi.cu` (not compiled here)

Implements Fig. 23.5-23.11's MPI-based multi-GPU Jacobi loop: each rank
owns a contiguous horizontal band of rows (1D domain decomposition along
y, chosen by the book because it keeps each partition's halo rows
contiguous in row-major memory -- §23.1) plus one halo row above and one
below. Each iteration: reset the L2-norm accumulator, launch the (unchanged
from file 01) Jacobi kernel over the whole local slab, then perform two
`MPI_Sendrecv` calls for the halo exchange (send top boundary row / receive
bottom halo row; send bottom boundary row / receive top halo row), then
`MPI_Allreduce` (`MPI_SUM`) the per-rank L2-norm-squared values and take the
square root on the host. Device pointers are passed directly to
`MPI_Sendrecv`/`MPI_Allreduce`, relying on CUDA-aware MPI, exactly as
§23.2 describes.

**Judgment call, clearly grounded in the text:** per §23.2's own words --
"the code uses a wrap-around strategy where the topmost rank treats the
bottommost rank as its top neighbor, and vice versa... often referred to as
the periodic boundary condition technique" -- the top/bottom rank-neighbor
topology is periodic (a torus in the y-dimension across all ranks), so
there is no true global top/bottom Dirichlet edge in this or any of
`03`-`05`, unlike file 01's single-GPU baseline (which has no ranks at all
and thus a genuine fixed top/bottom edge). The x-dimension is never
decomposed in the book, so `x=0`/`x=nx-1` remain true, non-periodic
Dirichlet edges in every file, consistent with file 01. Since the
y-topology is periodic and has no edge of its own, `02`-`05` all drive the
problem from the x-edge instead: `initGridKernel` sets `x=0` to `1.0`
(everything else, including `x=nx-1`, to `0.0`) -- the direct analog of
file 01's `y=0` edge, applied to the one axis that still has a true edge
in these files. Neither boundary column is ever written by the Jacobi
kernel (whose update only touches `0 < ix < nx-1`), so both stay fixed for
the life of the run, exactly like file 01's Dirichlet edges.

**Confidence:** high. §23.2's prose describes essentially every line of
Fig. 23.5-23.11 (the five bootstrap MPI calls, the `MPI_Send`/`MPI_Recv`/
`MPI_Sendrecv` signatures, the `MPI_Allreduce` signature, and the exact
send/receive row offsets used for the two `MPI_Sendrecv` calls), so this
file is a close reconstruction, not a guess. The actual figures render as
images in `pdftotext` output (as in earlier chapters, e.g. Ch. 21), so the
line-numbered code itself was not directly extractable -- reconstruction is
from the prose, which is unusually detailed for this section. As an extra
sanity check (not a substitute for compiling against the real MPI headers),
this file was compiled with `nvcc -c` against a hand-written stub `mpi.h`
reproducing the standard MPI-3 C signatures from memory; it compiled
without errors.

## §23.3 -- `03_stencil_multigpu_mpi_overlap.cu` (not compiled here)

Builds on `02` with the two-stage overlap strategy from Fig. 23.12-23.15:
the single Jacobi kernel launch is split into three launches of the *same*
kernel on three CUDA streams (`topStream`/`bottomStream` created with high
priority via `cudaStreamCreateWithPriority`, `internalStream` with low
priority) --

- `topStream`: `numRows=3` at offset 0 -> computes only the top boundary row
- `bottomStream`: `numRows=3` at offset `(nyLocal-3)*nx` -> computes only the bottom boundary row
- `internalStream`: `numRows=nyLocal-2` at offset `nx` -> computes the strictly-interior rows

Together these cover exactly the same rows as `02`'s single launch, with no
overlap and no gap. `cudaEvent`s (`cudaEventCreateWithFlags(...,
cudaEventDisableTiming)`) order the L2-norm reset before the boundary
kernels (`resetL2`) and order the L2-norm `cudaMemcpyAsync` (device to
pinned host memory, in `internalStream`) after both boundary kernels finish
(`topDone`/`bottomDone`), matching §23.3's description of
`cudaStreamWaitEvent`/`cudaEventRecord` usage. Per §23.3's explicitly
stated optimization ("we can use cudaMemcpyAsync to insert the memory copy
of the l2norm from the global memory to the host memory before the calls
to MPI_Sendrecv... Doing so overlaps the memory copy from the GPU to the
host CPU with the network communication"), this `cudaMemcpyAsync` is issued
*before* the two `MPI_Sendrecv` calls, not after -- since it is
non-blocking, the host proceeds immediately to `cudaStreamSynchronize` on
`topStream`/`bottomStream` and the two blocking `MPI_Sendrecv` calls while
the D2H copy (and the internal kernel, never synchronized on until the very
end) can still be in flight. A final `cudaStreamSynchronize(internalStream)`
after both `MPI_Sendrecv` calls ensures the copy has completed before
`l2norm_h` is read for `MPI_Allreduce`. `l2norm_h` is allocated with
`cudaMallocHost` (pinned memory), required because it is the target of
`cudaMemcpyAsync`, per §23.3's closing discussion of pinned memory and DMA.

**Confidence:** high for the stream/event/kernel-split structure (§23.3's
prose walks through Fig. 23.14 in detail, including exact row offsets and
even the exact `cudaStreamCreateWithPriority`/`cudaEventCreateWithFlags`
code snippets, which are given as inline code rather than an image figure).
Compiled cleanly with `nvcc -c` against the same MPI stub as `02`.

## §23.4 -- `04_stencil_multigpu_nccl.cu` (not compiled here)

Same three-stream structure as `03`, with the halo exchange replaced by
NCCL point-to-point calls placed directly in the streams instead of a
host-blocking `MPI_Sendrecv`:

- Bootstrap (Fig. 23.16-23.17): rank 0 calls `ncclGetUniqueId`, broadcasts
  the ID to all ranks via `MPI_Bcast`, all ranks `MPI_Barrier` then call
  `ncclCommInitRank` with their MPI rank as their NCCL rank (valid because
  there is a 1-to-1 rank-to-GPU mapping, as the book assumes).
- Halo exchange (Fig. 23.18-23.20): each former `MPI_Sendrecv` becomes an
  `ncclGroupStart()` / `ncclSend()` / `ncclRecv()` / `ncclGroupEnd()`
  group -- NCCL has no fused send-receive primitive, so the group call is
  used instead, per §23.4. The group replacing the top-boundary exchange
  runs in `topStream`; the group replacing the bottom-boundary exchange
  runs in `bottomStream`, matching "we pass the streams to ncclSend and
  ncclRecv as their last parameter."
- Two new events (`exchangeTop`, `exchangeBottom`) are recorded in
  `topStream`/`bottomStream` right after their NCCL group calls;
  `internalStream` waits on both before the L2-norm copy, replacing `03`'s
  host-side `cudaStreamSynchronize` + blocking `MPI_Sendrecv` entirely. No
  separate "kernel done" event is needed here (unlike `03`) because the
  NCCL calls are already stream-ordered after the boundary kernels in the
  same stream, so waiting on the exchange-complete event subsumes waiting
  for kernel completion -- this is a genuine, book-described simplification
  ("the host is now free from the communication and synchronization
  operations, except for the synchronization on the internal stream before
  the call to MPI_Allreduce").
- `MPI_Allreduce` is still used for the L2-norm collective -- NCCL replaces
  only the halo exchange, per §23.4 ("NCCL is not a complete replacement
  for MPI, but rather, a library that can be used in conjunction with
  MPI").

**Confidence:** high for the bootstrap sequence and the group-call
send/receive pattern (both are described in unusual prose detail,
including the exact rationale for `ncclGroupStart`/`ncclGroupEnd` and the
event-based replacement for host synchronization). Medium-high for the
exact NCCL C signatures (`ncclSend`/`ncclRecv`/`ncclGroupStart`/
`ncclGroupEnd`/`ncclCommInitRank`/`ncclGetUniqueId`/`ncclCommDestroy`),
which are reconstructed from general knowledge of the NCCL 2.x API rather
than quoted verbatim from the (image-rendered) Fig. 23.16/23.18/23.19 --
the book's prose confirms the *names* and *parameter meanings* of these
functions but the figures themselves did not extract as text. Compiled
cleanly with `nvcc -c` against a hand-written stub `nccl.h` built from
this same recollection; a reviewer with access to a real NCCL install
should double-check argument order and `ncclDataType_t`/`ncclComm_t`
usage against the installed `nccl.h`.

## §23.5 -- `05_stencil_multigpu_nvshmem.cu` (not compiled here)

Note: because `nvshmem_float_p` is called from device code, building this
file for real (unlike `02`-`04`) requires relocatable device code and
linking against the NVSHMEM device runtime, e.g. `nvcc -rdc=true
-lnvshmem_host -lnvshmem_device` (or `-lnvshmem`).

Halo exchange via NVSHMEM one-sided put, per Fig. 23.22-23.25:

- `nvshmem_malloc`/`nvshmem_free` allocate the input/output grids on the
  symmetric heap instead of `cudaMalloc`/`cudaFree` (§23.5: "All PEs must
  participate when calling these routines").
- The Jacobi kernel (`jacobiKernelNvshmem`, modeled on Fig. 23.23) takes
  `topPe`/`bottomPe` as extra parameters. After a thread computes its new
  value, if it is on the local top boundary row (`y==1`) it calls
  `nvshmem_float_p(&out[(ny-1)*nx+ix], newVal, topPe)` -- pushing the value
  directly into the *top PE's bottom halo row* at the symmetric offset; if
  on the local bottom boundary row (`y==ny-2`) it pushes into the *bottom
  PE's top halo row* the same way. This is a direct transcription of the
  book's own line-by-line description of Fig. 23.23 lines 15-20.
- Host setup (Fig. 23.24): `nvshmemx_init_attr_t attr; attr.mpi_comm =
  &mpiComm; nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);` --
  the book explicitly states `mpi_comm` "points to an MPI communicator
  object," confirming it is a pointer field, not a value field.
- Main loop (Fig. 23.25): unlike `03`/`04`, only **one** kernel launch on
  **one** stream is needed per iteration -- the put calls inside the kernel
  fuse computation and communication, so there is no need to split the
  kernel into boundary/internal launches at all. After the kernel,
  `nvshmemx_barrier_all_on_stream(stream)` is required before the L2 norm
  can be trusted, because kernel completion only guarantees the puts were
  *issued*, not that the data has *arrived* at the target PE (§23.5,
  explicit). `MPI_Allreduce` is still used for the L2-norm collective, same
  as `04`.

**Confidence:** high for the overall structure and the put-based kernel
logic (the book's prose is extremely explicit here, describing exactly
which threads put to which PE and why, and explicitly naming
`nvshmemx_barrier_all_on_stream` and its purpose). Medium for the exact
NVSHMEM C symbol names (`nvshmem_malloc`, `nvshmem_free`, `nvshmem_float_p`,
`nvshmem_my_pe`, `nvshmem_n_pes`, `nvshmemx_init_attr`,
`NVSHMEMX_INIT_WITH_MPI_COMM`, `nvshmemx_barrier_all_on_stream`) -- the
first four and the barrier function are named explicitly in the book text;
`nvshmem_my_pe`/`nvshmem_n_pes` are not directly quoted in the extracted
pages but are the standard OpenSHMEM/NVSHMEM PE-identification calls and
are needed to determine each PE's rank and the total PE count (the book's
Fig. 23.24 caption describes doing exactly this, without giving the two
function names verbatim). A reviewer with a real NVSHMEM install should
double-check these two names and the exact field layout of
`nvshmemx_init_attr_t` against the installed `nvshmem.h`/`nvshmemx.h`, and
should note that this file includes both headers (`<nvshmem.h>` and
`<nvshmemx.h>`) because the book says two header files are added (lines
02-03 of Fig. 23.24) -- the `x`-suffixed API surface used here
(`nvshmemx_init_attr_t`, `nvshmemx_init_attr`,
`nvshmemx_barrier_all_on_stream`) is conventionally declared in
`nvshmemx.h` in the real library. Compiled cleanly with `nvcc -c` against
hand-written stub `nvshmem.h`/`nvshmemx.h` headers built from this same
recollection.

## Common domain-decomposition parameters across `02`-`05`

All four multi-GPU files use `nx=130` (matching file 01's row width) and a
global interior height of `nyTotalInterior=512`, divided evenly across
`numRanks`/`nPes` (assumed to divide evenly -- not checked at runtime,
since these files are not executed here). Each rank's local slab is
`nyLocal = nyTotalInterior/numRanks + 2` rows (2 halo rows). This mirrors
file 01's grid shape and Jacobi math exactly, differing only in the halo
handling and communication mechanism described in each section above.
