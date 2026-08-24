# Chapter 4: Compute architecture and scheduling

Source: *Programming Massively Parallel Processors*, 5th ed., Ch. 4 (pp. 67-92).

| File | Book section | Technique |
|------|-------------|-----------|
| `01_control_divergence_demo.cu` | §4.5 | `divergentKernel` vs. `divergenceFreeKernel`: a warp-divergent `if (threadIdx.x % 2 == 0)` branch (Fig. 4.9) compared against a branchless, arithmetically-selected equivalent that computes the identical per-element result |
| `02_query_device_properties.cu` | §4.8 | `cudaGetDeviceCount` / `cudaGetDeviceProperties`, printing the `cudaDeviceProp` fields the section's listing walks through |

Most of the chapter -- §4.1 (SM/streaming-processor architecture), §4.2
(block-to-SM scheduling), §4.3 (`__syncthreads()` and transparent
scalability), §4.4 (warp partitioning and SIMD/SIMT hardware), §4.6 (warp
scheduling and latency tolerance), and §4.7 (resource partitioning and
occupancy) -- is conceptual background about how the hardware assigns
and schedules blocks, warps, and threads. None of these sections present
a standalone kernel listing of their own (§4.3's Fig. 4.4 is an example of
*incorrect* `__syncthreads()` usage, not a runnable kernel), so they are
summarized here rather than given a sample file:

- A grid's blocks are assigned to SMs whole (never split), which is what
  makes block-wide `__syncthreads()` and shared memory possible (§4.2).
- Because blocks can run in any relative order, the same kernel code
  scales transparently from a small GPU to a large one -- "transparent
  scalability" (§4.3).
- Each SM further partitions its resident blocks into 32-thread warps
  (consecutive `threadIdx` values, linearized row-major for multi-D
  blocks) and executes each warp SIMD/SIMT-style: one instruction
  fetched and issued per warp, applied to all 32 threads' data at once
  (§4.4).
- An SM keeps more warps resident than it can issue in a given cycle so
  that when one warp stalls on a long-latency operation (e.g. a global
  memory access), another ready warp can be issued instead -- "latency
  tolerance" / "latency hiding" via fine-grained multithreading (§4.6).
- How many blocks/warps can be simultaneously resident on an SM depends
  on the dynamic partitioning of registers, shared memory, and thread
  slots among them, which determines a kernel's *occupancy* (§4.7).

## §4.5 Control divergence -- `01_control_divergence_demo.cu`

§4.5 explains that SIMD/SIMT hardware executes all 32 threads of a warp
in lockstep. When threads within a warp take different control-flow
paths, the hardware makes one pass per distinct path, masking off the
threads not on that path (Fig. 4.9); this is *control divergence*, and
its cost is "the extra passes the hardware needs to take... as well as
the execution resources that are consumed by the inactive threads in
each pass." §4.5's own text illustrates a divergent, threadIdx-dependent
condition with `if(threadIdx.x > 2)`; this sample instead uses
`threadIdx.x % 2 == 0` (the condition §4.3's Fig. 4.4 uses), which is
also threadIdx-based and so, by §4.5's own criterion, is exactly the kind
of condition the section is about -- and since `threadIdx.x` parity
alternates on every thread, every warp that takes this branch is
guaranteed to diverge (a stronger guarantee than `> 2`, which only
diverges the warps straddling the threshold).

The sample runs two kernels over the same 1M-element input array:

- **`divergentKernel`** branches on `threadIdx.x % 2` and runs one of two
  different per-element formulas (4000 sequential multiplications by
  slightly different constants) depending on which side of the branch a
  thread takes. Since `blockDim.x` (256) is even, `threadIdx.x` parity
  and global-index parity coincide, so this applies formula A to every
  even-indexed element and formula B to every odd-indexed element.
- **`divergenceFreeKernel`** computes the exact same per-element mapping
  with no thread-index-dependent branch at all: it arithmetically selects
  *which single factor* to use (`isEven * FACTOR_A + (1 - isEven) *
  FACTOR_B`, where `isEven` is exactly `1.0f` or `0.0f`) once, before the
  loop, so the loop itself is a single multiply per iteration -- the same
  per-thread arithmetic as one branch of `divergentKernel` -- and every
  thread in every warp executes the identical instruction stream. This
  keeps the two kernels doing equivalent total work per thread, so any
  timing difference is attributable to the branch/pass structure, not to
  one kernel doing more arithmetic than the other.

Both formulas are pure repeated multiplication (never combined with an
add in the same expression), so there's no fused-multiply-add ambiguity
between host and device arithmetic, and both kernels' results are
expected to match a CPU reference applying the same two formulas by
index parity. **PASS/FAIL is decided purely by whether both kernels
agree with the CPU reference** (checked with `nearlyEqual`); both
kernels' timings are printed for information only.

Both kernels are launched once before the timed region (results
discarded) as a warm-up: the first time a kernel is touched, the driver
may need to JIT-compile the embedded PTX into SASS for the actual GPU
(this build targets `-arch=sm_75`), and that one-time cost has nothing to
do with control divergence -- without a warm-up it can dominate whichever
kernel happens to be launched first and produce a misleading result. With
the warm-up in place, on this repo's RTX 4090 (verified stable across
repeated runs, a fresh process each time, and even with a cleared
`~/.nv/ComputeCache` to force a cold JIT), `divergentKernel` measures
~0.21 ms versus `divergenceFreeKernel`'s ~0.11 ms for the same
4000-iteration-per-thread workload -- roughly 2x, matching the book's
model of a divergent warp needing two full passes versus one.

## §4.8 Querying device properties -- `02_query_device_properties.cu`

`cudaGetDeviceCount()` returns how many CUDA-capable devices are present,
and `cudaGetDeviceProperties()` fills in a `cudaDeviceProp` struct for a
given device index. The sample iterates every visible device and prints
the fields §4.8's listing calls out by name: `maxThreadsPerBlock`,
`multiProcessorCount` (SM count), `clockRate`, `maxThreadsDim[0..2]`,
`maxGridSize[0..2]`, `regsPerBlock`, and `warpSize`. It also prints a few
identifying fields not named in §4.8's text (device name, compute
capability, total global memory, shared memory per block) as extra
context. PASS means `cudaGetDeviceCount` succeeds with a device count
greater than 0 and `cudaGetDeviceProperties` succeeds for every device
found.

Build and run all samples in this chapter:

```sh
make run
```
