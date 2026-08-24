# PMPP CUDA Samples Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable CUDA C++ code repository derived from "Programming Massively Parallel Processors, 5th Edition" (Ch 2–23), organized to mirror the book's structure, with every sample compiled and verified against a CPU reference on real hardware.

**Architecture:** One git repo (`pmpp-cuda-samples/`), one folder per chapter under `part{1,2,3}-.../ch##-.../`. Each chapter folder is a self-contained set of numbered `.cu` files (one per kernel/algorithm variant the chapter presents), a `Makefile`, and a `README.md`. All files share one header-only utility, `common/cuda_utils.h`. Chapters have no dependencies on each other.

**Tech Stack:** CUDA Toolkit 12.6 (`nvcc`), GNU Make, C++17. Target GPUs: RTX 2070 SUPER (sm_75), RTX 4090 (sm_89), both physically present on this machine.

**Spec:** `docs/superpowers/specs/2026-08-24-pmpp-cuda-samples-design.md`

## Global Constraints

- Source PDF: `../Programming_Massively_Parallel_Processors_5E_-_Wen-mei_W_Hwu.pdf` (relative to repo root). Extract chapter text with `pdftotext -f <start> -l <end> -layout "<pdf>" -`.
- Scope: Chapters 2–23 only. Implement only code the chapter text actually presents (named kernels/host code). Do not implement end-of-chapter exercises. Do not invent examples not grounded in the extracted chapter text.
- Default compile target: `-arch=sm_75`. Exception: any file using `cuda::memcpy_async`/`cp.async` (Ch6 double buffering, Ch15 software pipelining) compiles with `-arch=sm_80` and only needs to run on the RTX 4090.
- Every `.cu` file must: generate its own test input, compute a CPU reference result, run the CUDA kernel(s), compare results (`nearlyEqual` for floats, exact for ints), and print `PASS`/`FAIL` plus a timing line. A task is not done until its binaries build clean and print `PASS` when run.
- `#include "../../common/cuda_utils.h"` (relative path from a `ch##-*/` folder) in every `.cu` file for `CUDA_CHECK`, `nearlyEqual`, and `GpuTimer`.
- No system-wide MPI/NCCL/NVSHMEM/cuDNN dev packages are installed on this machine (checked: only vendored copies inside unrelated Python venvs exist, which are not suitable to link against). Any sample needing them (Ch23 MPI/NCCL/NVSHMEM kernels, Ch19's optional cuDNN mention) is still written in full, but its Makefile target is guarded to skip the build with a printed message if the required header isn't found system-wide, and the chapter README states plainly that it wasn't compiled/run here.
- The repo is pushed to `git@github.com:rkaunismaa/Programming-Massively-Parallel-Processors-5th-Edition.git` (remote `origin`, branch `main`, already set up). The source PDF lives one directory above the repo root and must never be added to the git index.
- Commit after each task, then push immediately, with `git -C pmpp-cuda-samples add <paths> && git -C pmpp-cuda-samples -c user.email="kayintorob@yahoo.com" -c user.name="rob" commit -m "<message>" && git -C pmpp-cuda-samples push`. Each chapter is its own commit — never batch multiple chapters into one commit.

---

## Standard Makefile template

Used verbatim by every chapter task unless the task says otherwise (Ch6, Ch15 have an amended version — see their tasks).

```makefile
NVCC := nvcc
ARCH := -arch=sm_75
CFLAGS := -O2 -std=c++17 -I../../common
SRCS := $(wildcard *.cu)
BINS := $(patsubst %.cu,bin/%,$(SRCS))

all: $(BINS)

bin/%: %.cu | bin
	$(NVCC) $(ARCH) $(CFLAGS) -o $@ $<

bin:
	mkdir -p bin

run: all
	@for b in $(BINS); do echo "== $$b =="; ./$$b || exit 1; done

clean:
	rm -rf bin

.PHONY: all run clean
```

## Standard chapter README template

```markdown
# Chapter <N>: <Title>

Source: *Programming Massively Parallel Processors*, 5th ed., Ch. <N> (pp. <printed-range>).

| File | Book section | Technique |
|------|-------------|-----------|
| `01_....cu` | §<N.x> | <one-line description> |
| ... | | |

Build and run all samples in this chapter:

\`\`\`sh
make run
\`\`\`
```

---

### Task 0: Repo skeleton, shared utility header, top-level README

**Files:**
- Create: `pmpp-cuda-samples/common/cuda_utils.h`
- Create: `pmpp-cuda-samples/README.md`
- Create: `pmpp-cuda-samples/.gitignore`
- Create (empty dirs, via a placeholder or just created on first chapter): `part1-fundamental-concepts/`, `part2-parallel-patterns/`, `part3-advanced-patterns-and-applications/`

**Interfaces:**
- Produces: `CUDA_CHECK(call)` macro, `inline bool nearlyEqual(float a, float b, float eps = 1e-3f)`, class `GpuTimer` with `void start()` and `float stopAndGetMs()`. Every later task's `.cu` files consume these three names exactly.

- [ ] **Step 1: Write `common/cuda_utils.h`**

```cpp
#ifndef PMPP_CUDA_UTILS_H
#define PMPP_CUDA_UTILS_H

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                         \
        if (err__ != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,\
                    cudaGetErrorString(err__));                             \
            exit(EXIT_FAILURE);                                            \
        }                                                                   \
    } while (0)

inline bool nearlyEqual(float a, float b, float eps = 1e-3f) {
    float diff = fabsf(a - b);
    float largest = fmaxf(fabsf(a), fabsf(b));
    return diff <= eps * fmaxf(1.0f, largest);
}

class GpuTimer {
public:
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start() { CUDA_CHECK(cudaEventRecord(start_)); }
    float stopAndGetMs() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_, stop_;
};

#endif  // PMPP_CUDA_UTILS_H
```

- [ ] **Step 2: Write `.gitignore`**

```
bin/
*.o
*.out
```

- [ ] **Step 3: Write top-level `README.md`**

```markdown
# PMPP CUDA Samples

Unofficial, hand-written CUDA C++ code samples for every code-bearing
chapter (2–23) of *Programming Massively Parallel Processors: A
Hands-on Approach*, 5th Edition (Hwu, Kirk, El Hajj). The book does not
ship an official repository; this project builds one, organized to
mirror the book's own structure.

## Prerequisites

- CUDA Toolkit 12.6+ (`nvcc` on `PATH`)
- GNU Make
- An NVIDIA GPU, compute capability 7.5+ (sm_75). Two chapters (6 and
  15) have one sample each that requires compute capability 8.0+
  (Ampere or newer) — noted in their READMEs.

## Layout

- `part1-fundamental-concepts/` — Ch 2–6
- `part2-parallel-patterns/` — Ch 7–15
- `part3-advanced-patterns-and-applications/` — Ch 16–23
- `common/cuda_utils.h` — shared error-checking, timing, and
  float-comparison helpers used by every sample

## Build & run

Each chapter folder is independent:

\`\`\`sh
make -C part2-parallel-patterns/ch07-convolution run
\`\`\`

builds every sample in that chapter and runs it, printing PASS/FAIL and
timing for each.

## Scope

Samples implement the kernels, host code, and algorithms the book's
prose and figures actually present, chapter by chapter. End-of-chapter
exercises are not implemented. Appendices A–C and Chapter 1/24 are out
of scope (see the design spec in `docs/superpowers/specs/`).
```

- [ ] **Step 4: Create the three part directories so they're not empty**

```sh
cd pmpp-cuda-samples
mkdir -p part1-fundamental-concepts part2-parallel-patterns part3-advanced-patterns-and-applications
```

- [ ] **Step 5: Commit**

```sh
git add common README.md .gitignore
git -c user.email="kayintorob@yahoo.com" -c user.name="rob" commit -m "Add shared CUDA utils header and top-level README"
git push
```

---

## Chapter tasks (Tasks 1–22)

Each task below follows the same 6-step shape:

1. Extract the chapter's text from the PDF.
2. Write the numbered `.cu` files listed in the task (each self-contained: CPU reference, kernel, correctness check, timing, using `common/cuda_utils.h` per the Global Constraints).
3. Write the chapter's `Makefile` (standard template above, unless noted).
4. Build: `make -C <dir>` — must succeed with no errors.
5. Run: `make -C <dir> run` — every binary must print `PASS`.
6. Write the chapter's `README.md` (template above) and commit.

The file lists below are derived from the book's own section headings
(read from the table of contents) and are the intended one-kernel(-family)-per-file
breakdown. While extracting and reading the chapter text (step 1), confirm
each listed file still matches a real, distinct code listing in the
chapter; if a section turns out to be purely conceptual with no
listing, drop that file and say so in the README instead of inventing
code for it — do not pad the folder.

---

### Task 1: Chapter 2 — Heterogeneous data-parallel computing

**Files:**
- Create: `part1-fundamental-concepts/ch02-heterogeneous-data-parallel-computing/01_vector_addition.cu`
- Create: `.../ch02.../Makefile`, `.../ch02.../README.md`

**Interfaces:** Consumes `common/cuda_utils.h` (Task 0). Produces nothing consumed elsewhere.

- [ ] **Step 1:** `pdftotext -f 49 -l 72 -layout "../Programming_Massively_Parallel_Processors_5E_-_Wen-mei_W_Hwu.pdf" -` and read it.
- [ ] **Step 2:** Write `01_vector_addition.cu` covering §2.3–2.6: a CPU reference vector-add, `cudaMalloc`/`cudaMemcpy` host code (§2.4), the `vecAddKernel` (§2.5), and the kernel launch with a computed grid size (§2.6). Verify output against the CPU reference, print PASS/FAIL and GPU time via `GpuTimer`.
- [ ] **Step 3:** Write `Makefile` (standard template).
- [ ] **Step 4:** `make -C part1-fundamental-concepts/ch02-heterogeneous-data-parallel-computing` — must succeed.
- [ ] **Step 5:** `make -C part1-fundamental-concepts/ch02-heterogeneous-data-parallel-computing run` — must print PASS.
- [ ] **Step 6:** Write `README.md` (template), commit:
  `git add part1-fundamental-concepts/ch02-heterogeneous-data-parallel-computing && git commit -m "Add Ch2 vector addition sample" && git push`

---

### Task 2: Chapter 3 — Multidimensional grids and data

**Files:**
- Create: `part1-fundamental-concepts/ch03-multidimensional-grids-and-data/01_rgb_to_grayscale.cu` (§3.2)
- Create: `.../02_image_blur.cu` (§3.3)
- Create: `.../03_matrix_multiplication_naive.cu` (§3.4)
- Create: `.../Makefile`, `.../README.md`

- [ ] **Step 1:** `pdftotext -f 73 -l 94 -layout "<pdf>" -` and read it.
- [ ] **Step 2:** `01_rgb_to_grayscale.cu`: 2D-grid kernel mapping one thread per pixel converting an RGB buffer to grayscale (§3.2), with a synthetic test image and a CPU reference conversion.
- [ ] **Step 3:** `02_image_blur.cu`: 2D-grid blur kernel averaging a square neighborhood per pixel (§3.3), CPU reference blur, boundary handling for edge pixels.
- [ ] **Step 4:** `03_matrix_multiplication_naive.cu`: 2D-grid one-thread-per-output-element matmul (§3.4), CPU reference matmul.
- [ ] **Step 5:** Write `Makefile` (standard template).
- [ ] **Step 6:** `make -C part1-fundamental-concepts/ch03-multidimensional-grids-and-data run` — all three must PASS.
- [ ] **Step 7:** Write `README.md`, commit: `"Add Ch3 grayscale, blur, and naive matmul samples"`

---

### Task 3: Chapter 4 — Compute architecture and scheduling

**Files:**
- Create: `part1-fundamental-concepts/ch04-compute-architecture-and-scheduling/01_control_divergence_demo.cu` (§4.5)
- Create: `.../02_query_device_properties.cu` (§4.8)
- Create: `.../Makefile`, `.../README.md`

- [ ] **Step 1:** `pdftotext -f 95 -l 120 -layout "<pdf>" -` and read it.
- [ ] **Step 2:** `01_control_divergence_demo.cu`: two kernels operating on the same data — one with a warp-divergent branch (e.g. `if (threadIdx.x % 2 == 0)`), one with an equivalent divergence-free formulation as discussed in §4.5. "Correctness" here is that both kernels produce the same, CPU-verified result; report both kernels' timings so divergence's cost is visible in the output. Print PASS/FAIL based on result equality, not on which is faster.
- [ ] **Step 3:** `02_query_device_properties.cu`: call `cudaGetDeviceCount`/`cudaGetDeviceProperties` for every device and print the fields the book's §4.8 listing covers (max threads per block, SM count, warp size, shared mem per block, etc.). "PASS" here means the calls return `cudaSuccess` and device count > 0.
- [ ] **Step 4:** Write `Makefile` (standard template).
- [ ] **Step 5:** `make -C part1-fundamental-concepts/ch04-compute-architecture-and-scheduling run` — both must PASS.
- [ ] **Step 6:** Write `README.md`, commit: `"Add Ch4 control divergence and device query samples"`

---

### Task 4: Chapter 5 — Memory architecture and data locality

**Files:**
- Create: `part1-fundamental-concepts/ch05-memory-architecture-and-data-locality/01_tiled_matrix_multiplication.cu` (§5.4)
- Create: `.../02_tiled_matmul_boundary_checked.cu` (§5.5)
- Create: `.../Makefile`, `.../README.md`

- [ ] **Step 1:** `pdftotext -f 121 -l 150 -layout "<pdf>" -` and read it.
- [ ] **Step 2:** `01_tiled_matrix_multiplication.cu`: shared-memory tiled matmul (§5.4) for matrix dimensions that are exact multiples of the tile width, CPU reference matmul, PASS/FAIL + timing (and compare timing against a naive matmul computed in the same file for context).
- [ ] **Step 3:** `02_tiled_matmul_boundary_checked.cu`: same kernel generalized with the boundary checks from §5.5 for arbitrary (non-tile-multiple) matrix dimensions.
- [ ] **Step 4:** Write `Makefile` (standard template).
- [ ] **Step 5:** `make -C part1-fundamental-concepts/ch05-memory-architecture-and-data-locality run` — both must PASS.
- [ ] **Step 6:** Write `README.md`, commit: `"Add Ch5 tiled matmul samples"`

---

### Task 5: Chapter 6 — Performance considerations (has the sm_80 exception)

**Files:**
- Create: `part1-fundamental-concepts/ch06-performance-considerations/01_coalesced_vs_uncoalesced_access.cu` (§6.1)
- Create: `.../02_vectorized_loads_float4.cu` (§6.3)
- Create: `.../03_shared_memory_bank_conflicts.cu` (§6.4)
- Create: `.../04_thread_coarsening.cu` (§6.5)
- Create: `.../05_loop_unrolling.cu` (§6.6)
- Create: `.../06_double_buffering_async_copy.cu` (§6.7) — **sm_80 only**
- Create: `.../Makefile` (amended), `.../README.md`

Note: Ch6 lives under `part1-fundamental-concepts/` per the repo layout in the spec (Ch2–6 are Part 1), not `part2`.

- [ ] **Step 1:** `pdftotext -f 151 -l 186 -layout "<pdf>" -` and read it.
- [ ] **Step 2:** `01_coalesced_vs_uncoalesced_access.cu`: two kernels reading/writing the same array with a coalesced vs. strided/transposed access pattern (§6.1); both must produce CPU-verified-correct output, report both timings.
- [ ] **Step 3:** `02_vectorized_loads_float4.cu`: a `float4`-vectorized vector-add or similar elementwise kernel (§6.3) vs. a scalar version, both verified against CPU reference.
- [ ] **Step 4:** `03_shared_memory_bank_conflicts.cu`: a kernel accessing shared memory with a conflicting stride vs. a padded, conflict-free version (§6.4), both verified for correctness, timings reported.
- [ ] **Step 5:** `04_thread_coarsening.cu`: apply thread coarsening (§6.5) to the tiled matmul from Ch5 (re-implemented locally in this file — do not `#include` across chapter folders), compare against an uncoarsened version.
- [ ] **Step 6:** `05_loop_unrolling.cu`: the same or a similar tiled kernel with `#pragma unroll` applied to its inner loop (§6.6), compared against the unrolled-off version.
- [ ] **Step 7:** `06_double_buffering_async_copy.cu`: double-buffered shared-memory tile loading using `cuda::memcpy_async`/`cooperative_groups::memcpy_async` or raw `cp.async` PTX as described in §6.7, compiled with `-arch=sm_80`. CPU-verified correctness.
- [ ] **Step 8:** Write `Makefile`:

```makefile
NVCC := nvcc
CFLAGS := -O2 -std=c++17 -I../../common
SM75_SRCS := 01_coalesced_vs_uncoalesced_access.cu 02_vectorized_loads_float4.cu 03_shared_memory_bank_conflicts.cu 04_thread_coarsening.cu 05_loop_unrolling.cu
SM80_SRCS := 06_double_buffering_async_copy.cu
SM75_BINS := $(patsubst %.cu,bin/%,$(SM75_SRCS))
SM80_BINS := $(patsubst %.cu,bin/%,$(SM80_SRCS))
BINS := $(SM75_BINS) $(SM80_BINS)

all: $(BINS)

$(SM75_BINS): bin/%: %.cu | bin
	$(NVCC) -arch=sm_75 $(CFLAGS) -o $@ $<

$(SM80_BINS): bin/%: %.cu | bin
	$(NVCC) -arch=sm_80 $(CFLAGS) -o $@ $<

bin:
	mkdir -p bin

run: all
	@for b in $(BINS); do echo "== $$b =="; ./$$b || exit 1; done

clean:
	rm -rf bin

.PHONY: all run clean
```

- [ ] **Step 9:** `make -C part1-fundamental-concepts/ch06-performance-considerations` — must build clean (file 06 requires the 4090; if only the 2070S is visible to the shell, set `CUDA_VISIBLE_DEVICES` to select the 4090 for running, not building — building doesn't need a GPU).
- [ ] **Step 10:** `make -C part1-fundamental-concepts/ch06-performance-considerations run` — all six must PASS; run on the RTX 4090 (`CUDA_VISIBLE_DEVICES=<4090 index>`).
- [ ] **Step 11:** Write `README.md`, noting file 06 requires compute capability ≥ 8.0. Commit: `"Add Ch6 performance-consideration samples"`

---

### Task 6: Chapter 7 — Convolution

**Files:**
- Create: `part2-parallel-patterns/ch07-convolution/01_convolution_naive.cu` (§7.2)
- Create: `.../02_convolution_constant_memory.cu` (§7.4)
- Create: `.../03_convolution_tiled_halo.cu` (§7.5)
- Create: `.../04_convolution_tiled_cache_halo.cu` (§7.6)
- Create: `.../Makefile`, `.../README.md`

- [ ] **Step 1:** `pdftotext -f 187 -l 209 -layout "<pdf>" -` and read it. Confirm whether the chapter's running example is 1D or 2D convolution (the figure preview suggests it opens with 1D, §7.1) and use whichever the chapter's kernels (§7.2 onward) actually implement.
- [ ] **Step 2:** `01_convolution_naive.cu`: basic convolution kernel with the mask in global memory (§7.2), CPU reference, boundary (ghost-cell) handling as described.
- [ ] **Step 3:** `02_convolution_constant_memory.cu`: same kernel with the mask moved to `__constant__` memory (§7.4).
- [ ] **Step 4:** `03_convolution_tiled_halo.cu`: shared-memory tiled convolution that explicitly loads halo cells into shared memory (§7.5).
- [ ] **Step 5:** `04_convolution_tiled_cache_halo.cu`: tiled convolution that loads only the internal tile into shared memory and relies on the L2 cache for halo cells (§7.6).
- [ ] **Step 6:** Write `Makefile` (standard template).
- [ ] **Step 7:** `make -C part2-parallel-patterns/ch07-convolution run` — all four must PASS.
- [ ] **Step 8:** Write `README.md`, commit: `"Add Ch7 convolution samples"`

---

### Task 7: Chapter 8 — Stencil computation

**Files:**
- Create: `part2-parallel-patterns/ch08-stencil-computation/01_stencil_naive.cu` (§8.2)
- Create: `.../02_stencil_shared_memory_tiling.cu` (§8.4)
- Create: `.../03_stencil_thread_coarsening.cu` (§8.5)
- Create: `.../04_stencil_register_tiling.cu` (§8.6)
- Create: `.../Makefile`, `.../README.md`

- [ ] **Step 1:** `pdftotext -f 210 -l 227 -layout "<pdf>" -` and read it. Confirm the stencil's dimensionality/order (book historically uses a 3D 7-point stencil) from the actual text.
- [ ] **Step 2:** `01_stencil_naive.cu`: basic stencil sweep kernel (§8.2), CPU reference, boundary handling.
- [ ] **Step 3:** `02_stencil_shared_memory_tiling.cu`: shared-memory-tiled version (§8.4).
- [ ] **Step 4:** `03_stencil_thread_coarsening.cu`: thread-coarsened version, e.g. one thread sweeping a column of output points (§8.5).
- [ ] **Step 5:** `04_stencil_register_tiling.cu`: register-tiled version keeping the sweeping plane's values in registers instead of shared memory (§8.6).
- [ ] **Step 6:** Write `Makefile` (standard template).
- [ ] **Step 7:** `make -C part2-parallel-patterns/ch08-stencil-computation run` — all four must PASS.
- [ ] **Step 8:** Write `README.md`, commit: `"Add Ch8 stencil samples"`

---

### Task 8: Chapter 9 — Histogram

**Files:**
- Create: `part2-parallel-patterns/ch09-histogram/01_histogram_basic_atomics.cu` (§9.2)
- Create: `.../02_histogram_privatized_shared_mem.cu` (§9.4)
- Create: `.../03_histogram_coarsened.cu` (§9.5)
- Create: `.../04_histogram_thread_level_privatization.cu` (§9.6)
- Create: `.../Makefile`, `.../README.md`

- [ ] **Step 1:** `pdftotext -f 228 -l 247 -layout "<pdf>" -` and read it. Per the 5th-edition preface note, this chapter's histogram is computed over image data (e.g. a grayscale image's 256-bin intensity histogram) rather than text — confirm from the extracted text and use that.
- [ ] **Step 2:** `01_histogram_basic_atomics.cu`: direct `atomicAdd` into a global histogram (§9.2), CPU reference histogram.
- [ ] **Step 3:** `02_histogram_privatized_shared_mem.cu`: per-block private histogram in shared memory, merged into the global histogram (§9.4).
- [ ] **Step 4:** `03_histogram_coarsened.cu`: thread coarsening applied on top of privatization (§9.5).
- [ ] **Step 5:** `04_histogram_thread_level_privatization.cu`: per-thread private bins as described in §9.6.
- [ ] **Step 6:** Write `Makefile` (standard template).
- [ ] **Step 7:** `make -C part2-parallel-patterns/ch09-histogram run` — all four must PASS.
- [ ] **Step 8:** Write `README.md`, commit: `"Add Ch9 histogram samples"`

---

### Task 9: Chapter 10 — Reduction

**Files:**
- Create: `part2-parallel-patterns/ch10-reduction/01_reduction_naive.cu` (§10.3)
- Create: `.../02_reduction_reduced_divergence.cu` (§10.4)
- Create: `.../03_reduction_reduced_memory_divergence.cu` (§10.5)
- Create: `.../04_reduction_shared_memory.cu` (§10.6)
- Create: `.../05_reduction_warp_shuffle.cu` (§10.7)
- Create: `.../06_reduction_two_stage_warp_wide.cu` (§10.8)
- Create: `.../07_reduction_arbitrary_length.cu` (§10.9)
- Create: `.../08_reduction_thread_coarsening.cu` (§10.10)
- Create: `.../Makefile`, `.../README.md`

- [ ] **Step 1:** `pdftotext -f 248 -l 277 -layout "<pdf>" -` and read it.
- [ ] **Step 2–9:** Write each file above implementing the corresponding section's reduction-tree variant (sum reduction unless the text specifies otherwise), each with its own CPU reference sum and PASS/FAIL check. Each successive file should build on the technique named in its filename per the section it cites (§10.3 through §10.10 in order).
- [ ] **Step 10:** Write `Makefile` (standard template).
- [ ] **Step 11:** `make -C part2-parallel-patterns/ch10-reduction run` — all eight must PASS.
- [ ] **Step 12:** Write `README.md`, commit: `"Add Ch10 reduction samples"`

---

### Task 10: Chapter 11 — Scan

**Files:**
- Create: `part2-parallel-patterns/ch11-scan/01_scan_kogge_stone.cu` (§11.2)
- Create: `.../02_scan_kogge_stone_double_buffered.cu` (§11.3)
- Create: `.../03_scan_warp_level_primitives.cu` (§11.4)
- Create: `.../04_scan_coarsened.cu` (§11.6)
- Create: `.../05_scan_register_tiled.cu` (§11.7)
- Create: `.../06_scan_global_multi_block.cu` (§11.9)
- Create: `.../07_scan_brent_kung.cu` (§11.10)
- Create: `.../Makefile`, `.../README.md`

- [ ] **Step 1:** `pdftotext -f 278 -l 314 -layout "<pdf>" -` and read it.
- [ ] **Step 2–8:** Write each file implementing an inclusive (or exclusive, matching the book) prefix-sum scan per its cited section, each with a CPU reference scan and PASS/FAIL check.
- [ ] **Step 9:** Write `Makefile` (standard template).
- [ ] **Step 10:** `make -C part2-parallel-patterns/ch11-scan run` — all seven must PASS.
- [ ] **Step 11:** Write `README.md`, commit: `"Add Ch11 scan samples"`

---

### Task 11: Chapter 12 — Filter

**Files:**
- Create: `part2-parallel-patterns/ch12-filter/01_filter_unstable.cu` (§12.2)
- Create: `.../02_filter_warp_aggregated_atomics.cu` (§12.3)
- Create: `.../03_filter_privatized.cu` (§12.4)
- Create: `.../04_filter_stable.cu` (§12.5)
- Create: `.../05_filter_stable_coalesced_coarsened.cu` (§12.6)
- Create: `.../06_filter_in_place_stable.cu` (§12.7)
- Create: `.../Makefile`, `.../README.md`

- [ ] **Step 1:** `pdftotext -f 315 -l 328 -layout "<pdf>" -` and read it.
- [ ] **Step 2–7:** Write each file implementing stream-compaction ("filter elements matching a predicate") per its cited section: unstable/stable ordering, warp-aggregated atomics for the output-index counter, privatized counters, and in-place compaction, each with a CPU reference filter and a check that verifies the output set (and, for the "stable" variants, order) matches.
- [ ] **Step 8:** Write `Makefile` (standard template).
- [ ] **Step 9:** `make -C part2-parallel-patterns/ch12-filter run` — all six must PASS.
- [ ] **Step 10:** Write `README.md`, commit: `"Add Ch12 filter samples"`

---

### Task 12: Chapter 13 — Merge

**Files:**
- Create: `part2-parallel-patterns/ch13-merge/01_merge_basic_parallel.cu` (§13.4–13.5)
- Create: `.../02_merge_tiled.cu` (§13.6)
- Create: `.../03_merge_circular_buffer.cu` (§13.7)
- Create: `.../04_merge_coarsened.cu` (§13.8)
- Create: `.../Makefile`, `.../README.md`

- [ ] **Step 1:** `pdftotext -f 329 -l 354 -layout "<pdf>" -` and read it.
- [ ] **Step 2:** `01_merge_basic_parallel.cu`: implement the co-rank function (§13.4, both a `__host__` and `__device__` version) and the basic parallel merge kernel that uses it (§13.5) to merge two sorted arrays. CPU reference: sequential merge (§13.2).
- [ ] **Step 3:** `02_merge_tiled.cu`: shared-memory tiled merge kernel (§13.6).
- [ ] **Step 4:** `03_merge_circular_buffer.cu`: circular-buffer tiled merge kernel (§13.7).
- [ ] **Step 5:** `04_merge_coarsened.cu`: thread-coarsened merge (§13.8).
- [ ] **Step 6:** Write `Makefile` (standard template).
- [ ] **Step 7:** `make -C part2-parallel-patterns/ch13-merge run` — all four must PASS.
- [ ] **Step 8:** Write `README.md`, commit: `"Add Ch13 merge samples"`

---

### Task 13: Chapter 14 — Sorting

**Files:**
- Create: `part2-parallel-patterns/ch14-sorting/01_odd_even_sort.cu` (§14.2)
- Create: `.../02_merge_sort.cu` (§14.3)
- Create: `.../03_radix_sort_basic.cu` (§14.5)
- Create: `.../04_radix_sort_coalesced.cu` (§14.6)
- Create: `.../05_radix_sort_coarsened.cu` (§14.8)
- Create: `.../Makefile`, `.../README.md`

- [ ] **Step 1:** `pdftotext -f 355 -l 374 -layout "<pdf>" -` and read it.
- [ ] **Step 2:** `01_odd_even_sort.cu`: parallel odd-even transposition sort (§14.2), CPU reference `std::sort`, check full-array equality.
- [ ] **Step 3:** `02_merge_sort.cu`: parallel merge sort (§14.3) built from a merge kernel implemented locally in this file (do not cross-include Ch13's files).
- [ ] **Step 4:** `03_radix_sort_basic.cu`: parallel radix sort (§14.5) with a per-digit scan-based scatter.
- [ ] **Step 5:** `04_radix_sort_coalesced.cu`: radix sort optimized for coalesced global-memory access (§14.6).
- [ ] **Step 6:** `05_radix_sort_coarsened.cu`: thread-coarsened radix sort (§14.8).
- [ ] **Step 7:** Write `Makefile` (standard template).
- [ ] **Step 8:** `make -C part2-parallel-patterns/ch14-sorting run` — all five must PASS.
- [ ] **Step 9:** Write `README.md`, commit: `"Add Ch14 sorting samples"`

---

### Task 14: Chapter 15 — Advanced optimizations for matrix multiplication (has the sm_80 exception)

**Files:**
- Create: `part2-parallel-patterns/ch15-advanced-matmul-optimizations/01_matmul_coarsened_larger_tiles.cu` (§15.3)
- Create: `.../02_matmul_register_tiled.cu` (§15.4)
- Create: `.../03_matmul_coalesced_output_store.cu` (§15.5)
- Create: `.../04_matmul_bank_conflict_free.cu` (§15.6)
- Create: `.../05_matmul_software_pipelined.cu` (§15.8) — **sm_80 only**
- Create: `.../Makefile` (amended, same pattern as Task 5), `.../README.md`

- [ ] **Step 1:** `pdftotext -f 375 -l 398 -layout "<pdf>" -` and read it.
- [ ] **Step 2:** `01_matmul_coarsened_larger_tiles.cu`: matmul with coarsened threads computing larger output tiles (§15.3), CPU reference matmul.
- [ ] **Step 3:** `02_matmul_register_tiled.cu`: register-tiled input loading on top of the above (§15.4).
- [ ] **Step 4:** `03_matmul_coalesced_output_store.cu`: coalesced output-tile store (§15.5).
- [ ] **Step 5:** `04_matmul_bank_conflict_free.cu`: shared-memory layout that eliminates bank conflicts (§15.6).
- [ ] **Step 6:** `05_matmul_software_pipelined.cu`: `cuda::memcpy_async`/`cp.async`-based software pipelining (§15.8), compiled with `-arch=sm_80`.
- [ ] **Step 7:** Write `Makefile` using the same sm_75/sm_80-split pattern as Task 5's Ch6 Makefile, with `SM75_SRCS` = files 01–04 and `SM80_SRCS` = file 05.
- [ ] **Step 8:** `make -C part2-parallel-patterns/ch15-advanced-matmul-optimizations` — must build clean.
- [ ] **Step 9:** `make -C part2-parallel-patterns/ch15-advanced-matmul-optimizations run` — all five must PASS; run on the RTX 4090 (`CUDA_VISIBLE_DEVICES=<4090 index>`).
- [ ] **Step 10:** Write `README.md`, noting file 05 requires compute capability ≥ 8.0. Commit: `"Add Ch15 advanced matmul optimization samples"`

---

### Task 15: Chapter 16 — Dynamic programming and wavefront parallelism

**Files:**
- Create: `part3-advanced-patterns-and-applications/ch16-dynamic-programming-and-wavefront-parallelism/01_floyd_warshall.cu` (§16.4)
- Create: `.../02_smith_waterman.cu` (§16.5)
- Create: `.../03_wavefront_block_tiled.cu` (§16.6)
- Create: `.../04_hyperplane_transform.cu` (§16.7)
- Create: `.../Makefile`, `.../README.md`

- [ ] **Step 1:** `pdftotext -f 399 -l 426 -layout "<pdf>" -` and read it.
- [ ] **Step 2:** `01_floyd_warshall.cu`: parallel Floyd-Warshall all-pairs shortest path (§16.4), CPU reference Floyd-Warshall, compare distance matrices.
- [ ] **Step 3:** `02_smith_waterman.cu`: parallel Smith-Waterman local sequence alignment (§16.5), CPU reference DP, compare score matrices (and/or optimal score).
- [ ] **Step 4:** `03_wavefront_block_tiled.cu`: block-level tiling applied to whichever of the two DP kernels above the chapter uses for this section (§16.6) — confirm from the text which one, and reimplement that kernel's tiled version locally in this file.
- [ ] **Step 5:** `04_hyperplane_transform.cu`: the hyperplane-transformed (coordinate-remapped) wavefront kernel (§16.7).
- [ ] **Step 6:** Write `Makefile` (standard template).
- [ ] **Step 7:** `make -C part3-advanced-patterns-and-applications/ch16-dynamic-programming-and-wavefront-parallelism run` — all four must PASS.
- [ ] **Step 8:** Write `README.md`, commit: `"Add Ch16 dynamic programming and wavefront samples"`

---

### Task 16: Chapter 17 — Sparse matrix computation

**Files:**
- Create: `part3-.../ch17-sparse-matrix-computation/01_spmv_coo.cu` (§17.2)
- Create: `.../02_spmv_csr.cu` (§17.3)
- Create: `.../03_spmv_ell.cu` (§17.4)
- Create: `.../04_spmv_hybrid_ell_coo.cu` (§17.5)
- Create: `.../05_spmv_jds.cu` (§17.6)
- Create: `.../06_spmv_csc.cu` (§17.7)
- Create: `.../Makefile`, `.../README.md`

- [ ] **Step 1:** `pdftotext -f 427 -l 450 -layout "<pdf>" -` and read it.
- [ ] **Step 2–7:** For each file, implement sparse matrix–vector multiplication (SpMV) using the named storage format (COO, CSR, ELL, hybrid ELL-COO, JDS, CSC), each building its own small sparse test matrix (dense-equivalent known to the CPU reference), converting to that format on the host, running the CUDA kernel, and comparing against a dense CPU matvec. For CSC (§17.7 — column-wise access), implement whatever operation the section actually demonstrates (may be SpMV via the transpose, or a column-oriented op); confirm from the text.
- [ ] **Step 8:** Write `Makefile` (standard template).
- [ ] **Step 9:** `make -C part3-advanced-patterns-and-applications/ch17-sparse-matrix-computation run` — all six must PASS.
- [ ] **Step 10:** Write `README.md`, commit: `"Add Ch17 sparse matrix (SpMV) samples"`

---

### Task 17: Chapter 18 — Graph traversal

**Files:**
- Create: `part3-.../ch18-graph-traversal/01_bfs_vertex_centric.cu` (§18.3)
- Create: `.../02_bfs_edge_centric.cu` (§18.4)
- Create: `.../03_bfs_frontier_based.cu` (§18.5)
- Create: `.../04_bfs_frontier_privatized.cu` (§18.6)
- Create: `.../05_bfs_cooperative_groups.cu` (§18.7)
- Create: `.../Makefile`, `.../README.md`

- [ ] **Step 1:** `pdftotext -f 451 -l 477 -layout "<pdf>" -` and read it.
- [ ] **Step 2–6:** For each file, implement BFS from a fixed source vertex on a small synthetic graph (CSR adjacency), producing a level/distance array, checked against a CPU BFS reference: vertex-centric push (§18.3), edge-centric (§18.4), frontier-based work-efficient BFS (§18.5), privatized-frontier BFS (§18.6), and a grid-wide-synchronizing version using cooperative groups (§18.7) — cooperative-groups grid sync requires launching with `cudaLaunchCooperativeKernel`; check `cudaDeviceGetAttribute(cudaDevAttrCooperativeLaunch, ...)` and skip gracefully with a clear message if unsupported (it is supported on both local GPUs, so this should run in practice).
- [ ] **Step 7:** Write `Makefile` (standard template).
- [ ] **Step 8:** `make -C part3-advanced-patterns-and-applications/ch18-graph-traversal run` — all five must PASS.
- [ ] **Step 9:** Write `README.md`, commit: `"Add Ch18 graph traversal (BFS) samples"`

---

### Task 18: Chapter 19 — Convolutional neural networks

**Files:**
- Create: `part3-.../ch19-convolutional-neural-networks/01_cnn_conv_layer_direct.cu` (§19.2)
- Create: `.../02_cnn_conv_layer_as_gemm.cu` (§19.3)
- Create: `.../Makefile`, `.../README.md`

- [ ] **Step 1:** `pdftotext -f 478 -l 500 -layout "<pdf>" -` and read it.
- [ ] **Step 2:** `01_cnn_conv_layer_direct.cu`: direct CUDA convolutional-layer kernel (multi-channel input, multi-filter, as in §19.2), CPU reference conv layer.
- [ ] **Step 3:** `02_cnn_conv_layer_as_gemm.cu`: im2col-style unfolding of the input followed by a GEMM to compute the same convolutional layer (§19.3), checked against the same CPU reference as file 01.
- [ ] **Step 4:** §19.4 (cuDNN) is a library-usage discussion, not a from-scratch kernel, and no system cuDNN dev package is installed here — do not add a cuDNN sample; mention this in the README instead.
- [ ] **Step 5:** Write `Makefile` (standard template).
- [ ] **Step 6:** `make -C part3-advanced-patterns-and-applications/ch19-convolutional-neural-networks run` — both must PASS.
- [ ] **Step 7:** Write `README.md`, commit: `"Add Ch19 CNN convolution layer samples"`

---

### Task 19: Chapter 20 — Large language models

**Files:**
- Create: `part3-.../ch20-large-language-models/01_attention_naive.cu` (§20.3)
- Create: `.../02_attention_kv_cache.cu` (§20.4)
- Create: `.../03_flash_attention.cu` (§20.5)
- Create: `.../Makefile`, `.../README.md`

- [ ] **Step 1:** `pdftotext -f 501 -l 536 -layout "<pdf>" -` and read it.
- [ ] **Step 2:** `01_attention_naive.cu`: single-head scaled dot-product attention kernel(s) (QK^T, softmax, ·V) for a small sequence length (§20.3), CPU reference attention (in double or careful float precision), `nearlyEqual` comparison with a slightly looser epsilon given softmax's accumulated rounding.
- [ ] **Step 3:** `02_attention_kv_cache.cu`: incremental (autoregressive, one new token at a time) attention using a growing KV cache (§20.4), checked against full recomputation (file 01's approach) for equivalence at each step.
- [ ] **Step 4:** `03_flash_attention.cu`: tiled, online-softmax flash-attention kernel (§20.5) that avoids materializing the full attention matrix, checked against the naive attention output from file 01.
- [ ] **Step 5:** §20.7 (alleviating memory requirements, e.g. multi/grouped-query attention) — only add a fourth file if the extracted text contains an actual code listing for it; otherwise cover it in the README as a concept with no separate sample.
- [ ] **Step 6:** Write `Makefile` (standard template).
- [ ] **Step 7:** `make -C part3-advanced-patterns-and-applications/ch20-large-language-models run` — all files must PASS.
- [ ] **Step 8:** Write `README.md`, commit: `"Add Ch20 attention/LLM samples"`

---

### Task 20: Chapter 21 — Electrostatic potential map

**Files:**
- Create: `part3-.../ch21-electrostatic-potential-map/01_dcs_scatter.cu` (§21.2, scatter)
- Create: `.../02_dcs_gather.cu` (§21.2, gather)
- Create: `.../03_dcs_coarsened.cu` (§21.3)
- Create: `.../04_dcs_coalesced.cu` (§21.4)
- Create: `.../05_dcs_cutoff_binning.cu` (§21.5)
- Create: `.../Makefile`, `.../README.md`

- [ ] **Step 1:** `pdftotext -f 537 -l 552 -layout "<pdf>" -` and read it.
- [ ] **Step 2:** `01_dcs_scatter.cu` / `02_dcs_gather.cu`: direct Coulomb summation over a synthetic set of charged atoms onto a grid, implemented both as scatter (each atom writes to many grid points, needs atomics) and gather (each grid point reads all atoms) per §21.2, both checked against the same O(N·M) CPU reference.
- [ ] **Step 3:** `03_dcs_coarsened.cu`: gather-based kernel with thread coarsening across grid points (§21.3).
- [ ] **Step 4:** `04_dcs_coalesced.cu`: memory-coalescing-optimized layout on top of the coarsened kernel (§21.4).
- [ ] **Step 5:** `05_dcs_cutoff_binning.cu`: spatial binning with a cutoff radius so only nearby atoms are summed per grid point (§21.5), checked against the CPU reference restricted to the same cutoff.
- [ ] **Step 6:** Write `Makefile` (standard template).
- [ ] **Step 7:** `make -C part3-advanced-patterns-and-applications/ch21-electrostatic-potential-map run` — all five must PASS.
- [ ] **Step 8:** Write `README.md`, commit: `"Add Ch21 electrostatic potential map samples"`

---

### Task 21: Chapter 22 — Algorithm selection, problem decomposition, and problem formulation

**Files:** determined after reading the chapter (see Step 1) — this chapter is design-methodology focused (algorithm selection, Amdahl's law, batching latency vs. throughput), not kernel-by-kernel like the others.

- [ ] **Step 1:** `pdftotext -f 553 -l 564 -layout "<pdf>" -` and read it in full.
- [ ] **Step 2:** If the chapter contains at least one concrete, distinct code listing (e.g., a batching-vs-single-request throughput comparison for §22.5, or a decomposition example), create `part3-advanced-patterns-and-applications/ch22-algorithm-selection-and-problem-decomposition/0N_<name>.cu` implementing it, self-contained per the Global Constraints (own CPU reference/expected behavior, PASS/FAIL, timing where relevant).
- [ ] **Step 3:** If the chapter is entirely conceptual with no distinct runnable listing, do not create any `.cu` file. Instead create only `part3-advanced-patterns-and-applications/ch22-algorithm-selection-and-problem-decomposition/README.md` summarizing the chapter's design principles (algorithm selection criteria, decomposition strategies, Amdahl's law application, problem formulation, batching trade-offs) in your own words, with a note explaining why no code sample was added.
- [ ] **Step 4:** If any `.cu` files were created, write a `Makefile` (standard template) and run `make -C part3-advanced-patterns-and-applications/ch22-algorithm-selection-and-problem-decomposition run`, confirming PASS.
- [ ] **Step 5:** Commit: `git add part3-advanced-patterns-and-applications/ch22-algorithm-selection-and-problem-decomposition && git commit -m "Add Ch22 algorithm selection notes" && git push` (message becomes `"...samples"` if code was added).

---

### Task 22: Chapter 23 — Multi-GPU programming

**Files:**
- Create: `part3-.../ch23-multi-gpu-programming/01_stencil_singlegpu_baseline.cu` (§23.1) — always built/run
- Create: `.../02_stencil_multigpu_mpi.cu` (§23.2) — MPI, not built here
- Create: `.../03_stencil_multigpu_mpi_overlap.cu` (§23.3) — MPI, not built here
- Create: `.../04_stencil_multigpu_nccl.cu` (§23.4) — NCCL, not built here
- Create: `.../05_stencil_multigpu_nvshmem.cu` (§23.5) — NVSHMEM, not built here
- Create: `.../Makefile` (amended), `.../README.md`

- [ ] **Step 1:** `pdftotext -f 565 -l 600 -layout "<pdf>" -` and read it.
- [ ] **Step 2:** `01_stencil_singlegpu_baseline.cu`: the single-GPU Jacobi/stencil kernel used as the chapter's running example (§23.1), CPU reference, PASS/FAIL. This one must build and run on this machine (no external libs needed) — use it as the correctness baseline the other files' *logic* is modeled on, even though they won't be executed here.
- [ ] **Step 3:** `02_stencil_multigpu_mpi.cu` / `03_stencil_multigpu_mpi_overlap.cu`: full MPI-based multi-GPU domain-decomposed stencil (halo exchange via `MPI_Sendrecv` or similar) and its computation/communication-overlapped version (CUDA streams + MPI), written faithfully per §23.2–23.3, `#include <mpi.h>`.
- [ ] **Step 4:** `04_stencil_multigpu_nccl.cu`: same stencil using NCCL collectives/point-to-point for halo exchange (§23.4), `#include <nccl.h>`.
- [ ] **Step 5:** `05_stencil_multigpu_nvshmem.cu`: same stencil using NVSHMEM one-sided put/get for halo exchange (§23.5), `#include <nvshmem.h>`.
- [ ] **Step 6:** Write `Makefile`:

```makefile
NVCC := nvcc
ARCH := -arch=sm_75
CFLAGS := -O2 -std=c++17 -I../../common

all: bin/01_stencil_singlegpu_baseline

bin/01_stencil_singlegpu_baseline: 01_stencil_singlegpu_baseline.cu | bin
	$(NVCC) $(ARCH) $(CFLAGS) -o $@ $<

bin:
	mkdir -p bin

run: all
	@echo "== bin/01_stencil_singlegpu_baseline =="
	@./bin/01_stencil_singlegpu_baseline
	@echo "NOTE: 02-05 require MPI/NCCL/NVSHMEM, not installed on this machine; not built. See README."

clean:
	rm -rf bin

.PHONY: all run clean
```

- [ ] **Step 7:** `make -C part3-advanced-patterns-and-applications/ch23-multi-gpu-programming run` — file 01 must PASS.
- [ ] **Step 8:** Write `README.md` per the standard template, with an explicit note under the table: "`02`–`05` require MPI / NCCL / NVSHMEM respectively, none of which have system-wide dev packages installed on this machine; they are written to match the book's listings but were not compiled or run here." Commit: `"Add Ch23 multi-GPU programming samples"`

---

## Self-review notes

- **Spec coverage:** Every in-scope chapter (2–23) has a task; Task 0 covers the spec's shared-utility and top-level-README requirements; Tasks 5 and 14 implement the spec's sm_80 exception; Task 22 implements the spec's "skip gracefully, note in README" requirement for missing multi-GPU libraries.
- **Placeholder scan:** No task leaves a step as "TODO"/"handle appropriately" — every step names the specific algorithm, section, and check. Task 21 (Ch22) and the cuDNN note in Task 18 (Ch19) are deliberate scope narrowings grounded in the spec ("do not invent examples"), not placeholders.
- **Type/interface consistency:** All chapter tasks consume the exact three names Task 0 produces (`CUDA_CHECK`, `nearlyEqual`, `GpuTimer`) via the same relative include path `../../common/cuda_utils.h`. No task produces an interface another chapter task consumes (chapters are independent, as the spec requires), so there's no cross-task signature drift to check.
