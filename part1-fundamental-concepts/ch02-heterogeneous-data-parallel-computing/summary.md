# Chapter 2 — Heterogeneous Data-Parallel Computing

Section-by-section summary of the book content. Page numbers are printed
book pages (Chapter 2 runs pp. 21–43).

## 2.1 Data parallelism (p. 21)

**Data parallelism** is the phenomenon that computations on different
parts of a dataset can be done independently of each other, and so can
be done in parallel. It's distinct from **task parallelism** (different,
independent *tasks* — e.g. a vector addition and a matrix-vector
multiply in the same program — run concurrently); the two aren't
mutually exclusive, but data parallelism is the chapter's — and the
book's — main source of scalable speedup, because large datasets give
massively parallel hardware abundant independent work to do, and that
abundance grows automatically with dataset size.

Running example: converting a color image to grayscale (Fig. 2.1),
using the weighted-sum formula `L = r*0.299 + g*0.587 + b*0.114`
(Eq. 2.1) on each pixel's RGB triple (sidebar: RGB images are stored as
one `(r,g,b)` tuple per pixel). Since each output pixel `O[i]` depends
only on its own input pixel `I[i]` (Fig. 2.2), every pixel's computation
is independent of every other's — a textbook case of rich data
parallelism, since real images commonly have millions of pixels. Even
an apparently "global" operation (e.g. average brightness across a
whole image) can typically be decomposed into many independent partial
computations. Writing data-parallel code means (re)organizing a
computation around the data so independent pieces can be identified and
executed concurrently.

## 2.2 CUDA C++ program structure (p. 24)

A CUDA C++ source file can freely mix **host code** (runs on the CPU)
and **device code** (runs on the GPU); without any CUDA keywords, a
file is ordinary host-only C++. Device code lives in **kernels** —
functions marked with special CUDA keywords — and executes in a
data-parallel manner.

Execution starts on the host as ordinary sequential code (Fig. 2.3). A
kernel call launches a large number of threads on the device,
collectively called a **grid**, organized as multiple **blocks** of
threads; when all threads of a grid finish, the grid terminates and
control returns to the host, which may launch further grids — letting
heterogeneous applications overlap CPU and GPU work across multiple
kernel launches. CUDA follows the well-known **SPMD** (Single-Program,
Multiple-Data) style: every thread in a grid executes the *same* kernel
code, differentiated only by which data it operates on. Generating and
scheduling a CUDA thread costs very few clock cycles thanks to
dedicated hardware support — in sharp contrast to traditional CPU
threads, which can take thousands of cycles to create — which is what
makes launching millions of lightweight threads for a single kernel
practical.

## 2.3 A vector addition example (p. 26)

Introduces vector addition as the data-parallel "Hello World." A
conventional (host-only) C++ version (Fig. 2.4) is a simple `for` loop
over `n` elements, `C_h[i] = A_h[i] + B_h[i]`. The book's naming
convention, used throughout: variables holding host data are suffixed
`_h`, and variables holding device data `_d`.

The chapter revises `vecAdd` into three parts (Fig. 2.5, outlined as
comments to be filled in over the following sections): **Part 1**
allocates device memory for `A`, `B`, `C` and copies `A`/`B` from host
to device; **Part 2** calls a kernel to launch a grid of threads that
performs the addition on the device; **Part 3** copies the result `C`
back to host memory and frees the device memory. This "transparent
outsourcing" pattern — where the host function's signature and
semantics look unchanged, but the work now silently happens on a device
— is simple but can be inefficient in practice if done naively for
every call (every call pays full copy-out/copy-in cost); real
applications more often keep important data resident on the device
across multiple kernel calls. The rest of the chapter fills in Parts
1–3.

## 2.4 Device global memory and data transfer (p. 28)

CUDA devices typically have their own on-board DRAM, called **device
global memory** (or just **global memory**) — e.g. an NVIDIA Hopper
H100 ships with 80 or 94 GB. Implementing Part 1/3 of `vecAdd` needs
two CUDA runtime API functions:

- **`cudaMalloc(&ptr, size)`** — allocates `size` bytes in device
  global memory, writing the address into the pointer whose *address*
  is passed in (hence `(void**)&A_d`, since the function needs to set
  the pointer itself, mirroring C's `malloc` but with the destination
  pointer passed by address rather than returned, so the function's
  own return value stays free for error-reporting).
- **`cudaFree(ptr)`** — frees a device global-memory allocation,
  mirroring C's `free`.

Device pointers (`A_d`, `B_d`, `C_d`) must never be dereferenced in
host code — they refer to memory the host can't directly access, and
doing so causes a runtime error.

Data transfer between host and device uses **`cudaMemcpy(dst, src,
size, kind)`**, where `kind` is one of `cudaMemcpyHostToDevice` or
`cudaMemcpyDeviceToHost` (also `HostToHost`/`DeviceToDevice`, not used
here). Fig. 2.8 assembles the complete host-side `vecAdd` stub: `cudaMalloc`
for all three arrays, `cudaMemcpy` of `A`/`B` host→device, the (not yet
shown) kernel call, `cudaMemcpy` of `C` device→host, then `cudaFree`
for all three. A sidebar stresses checking every CUDA API call's
returned `cudaError_t` (via `cudaGetErrorString`) rather than
assuming success, since silent failures are hard to debug after the
fact.

## 2.5 Kernel functions and threading (p. 32)

A kernel launch produces a **grid** organized as a 2-level hierarchy:
a grid is an array of equally-sized **blocks**, and each block is an
array of threads (up to 1024 threads per block on current systems).
Each thread carries two built-in, read-only coordinate variables:
**`threadIdx`** (its index within its own block) and **`blockIdx`**
(its block's index within the grid) — both `dim3` structs with `.x`,
`.y`, `.z` fields, used for 1D/2D/3D organization matching however many
dimensions the data naturally has. A third built-in, **`blockDim`**,
holds the (uniform, grid-wide) size of each block. The book's analogy
(sidebar): `blockIdx` is like a telephone area code and `threadIdx` like
the local number — together they form one globally unique address,
and most work happens "locally" within a block the same way most calls
stay within an area code.

For a 1D vector-length problem, each thread computes its own unique,
contiguous global index as `i = blockIdx.x * blockDim.x +
threadIdx.x` (Fig. 2.9) — consecutive threads in a block get
consecutive `threadIdx.x` values, and consecutive blocks cover
consecutive, non-overlapping ranges of `i`, so the whole grid's threads
jointly cover a contiguous range with no gaps and no overlap.

The kernel itself (Fig. 2.10, `vecAddKernel`) is marked with the
**`__global__`** qualifier, computes its own `i`, and guards the actual
add with **`if (i < n)`** — necessary because block size doesn't
generally divide the data length evenly (e.g. 100 elements with
block size 32 needs 4 blocks = 128 threads, so the last 28 threads
must do nothing). Every automatic (local) variable such as `i` is
private per-thread — a grid of 10,000 threads creates 10,000 separate
copies of `i`, invisible to other threads. Compared to the host-only
Fig. 2.4, the kernel's explicit loop has disappeared entirely: the
whole *grid* now plays the role the loop used to play, with each
thread handling one loop iteration in parallel (**loop parallelism**).

Fig. 2.11 summarizes CUDA's three function-declaration qualifiers:
**`__host__`** (default if none given; callable only from host, runs
on host), **`__global__`** (callable from host, or from device given
dynamic parallelism support; runs on device as a kernel, launching a
new grid of threads), and **`__device__`** (callable only from device
code, runs on device, does not launch new threads — just an ordinary
function call on whichever thread invokes it). A function can combine
`__host__` and `__device__` to have the compiler generate both a host
and a device version of the same source.

## 2.6 Calling kernel functions (p. 37)

A kernel call uses ordinary C++ function-call syntax plus an
**execution configuration** in `<<<...>>>` brackets before the
argument list: `kernel<<<numBlocks, blockSize>>>(args)`, where the
first parameter gives the grid's block count and the second each
block's thread count (Fig. 2.12). To guarantee enough threads to cover
`n` elements regardless of `n`'s exact value, the block count is
computed with a **ceiling division**, `ceil(n/256.0)` for a block size
of 256 (dividing by `256.0`, a float, so the ceiling rounds correctly) —
this is also exactly why the kernel needs its `if (i < n)` guard, since
ceiling division can launch a few more threads than there are elements.
Fig. 2.13 shows the complete `vecAdd` host function: allocate, copy in,
launch with the computed grid/block dimensions, copy out, free — the
first full realization of the three-part skeleton from §2.3. Blocks of
a grid may execute in any relative order and on however many execution
units the device has at once, which is exactly what gives CUDA
**transparent scalability** across devices of different sizes (explored
fully in Chapter 4) — the same compiled kernel runs correctly, just at
different speed, whether the underlying GPU can run 1–2 blocks
concurrently or hundreds.

## 2.7 Compilation (p. 39)

CUDA C++'s extensions aren't valid standard C++, so compilation goes
through **NVCC** (NVIDIA CUDA Compiler, Fig. 2.14), which separates a
source file's host code from its device code: host code is compiled
by an ordinary host C/C++ compiler and runs as a normal CPU process;
device code (kernels, device functions, and the data structures they
use) is compiled by NVCC into an intermediate virtual binary format
called **PTX**, which a device-side just-in-time compiler further
compiles into real machine code and executes on the GPU at run time.

## 2.8 Summary (p. 40)

Recaps the chapter's four categories of CUDA C++ extension: **function
declarations** (`__global__`/`__device__`/`__host__`, Fig. 2.11);
**kernel call and grid launch** syntax (the `<<<...>>>` execution
configuration); **built-in variables** (`threadIdx`, `blockDim`,
`blockIdx`) that let each thread find its own data; and the **runtime
API** (`cudaMalloc`, `cudaFree`, `cudaMemcpy`) for managing device
memory and host↔device transfers. The chapter is explicitly a
simplified introduction, not a comprehensive CUDA reference — further
features are introduced progressively throughout the book as needed,
and the reader is pointed to the official CUDA C++ Programming Guide
for complete API details.

## 2.9 Exercises (p. 41)

Ten problems applying the chapter's concepts by hand: writing the
`threadIdx`/`blockIdx` index expression for various per-thread
work assignments, including multiple elements per thread (Q1–3);
computing total grid thread count for a given vector length and block
size (Q4); correct `cudaMalloc` argument expressions (Q5–6); a correct
`cudaMemcpy` call (Q7); how to declare a variable to hold a CUDA API
call's returned error code (Q8); reading off thread/block counts and
divergent-code-path participation from a given kernel + launch
configuration (Q9); and advice for a programmer frustrated by needing
to declare a function for both host and device twice (Q10 — answer:
use combined `__host__ __device__`). Not implemented in this repo's
samples (end-of-chapter exercises are out of scope — see the repo root
README).
