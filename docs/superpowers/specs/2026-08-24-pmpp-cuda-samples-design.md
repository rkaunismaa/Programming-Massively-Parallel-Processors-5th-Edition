# PMPP CUDA Samples — Design Spec

Date: 2026-08-24

## Purpose

"Programming Massively Parallel Processors, 5th Edition" (Hwu, Kirk, El Hajj)
does not ship an official code repository. This project builds one: a
runnable CUDA C++ code repository derived from the book's chapter content,
organized to mirror the book's structure, so the code from each chapter can
be read, built, and run independently.

Source: `Programming_Massively_Parallel_Processors_5E_-_Wen-mei_W_Hwu.pdf`
(674 PDF pages; PDF page = printed page + 26, with drift — exact chapter
page ranges below were located by searching for chapter-opener markers,
not by fixed offset).

## Scope

- Chapters 2–23 only (Chapter 1 is non-technical background; Chapter 24 is
  a conclusion with no code; Appendices A–C are conceptual/math or
  sequential-C++ reference material with negligible CUDA content).
- Only code that the chapter text actually presents or clearly specifies
  (named kernels, host drivers, data structures). End-of-chapter exercises
  are NOT implemented — those are reader assignments, not chapter content.
- One self-contained CUDA repo, git-initialized, committed locally.
  No GitHub remote is created (per user decision).

## Chapter → PDF page map

(1-indexed PDF pages, inclusive ranges, located via chapter-opener search)

| Ch | Title | PDF pages |
|----|-------|-----------|
| 2  | Heterogeneous data-parallel computing | 49–72 |
| 3  | Multidimensional grids and data | 73–94 |
| 4  | Compute architecture and scheduling | 95–120 |
| 5  | Memory architecture and data locality | 121–150 |
| 6  | Performance considerations | 151–186 |
| 7  | Convolution | 187–209 |
| 8  | Stencil computation | 210–227 |
| 9  | Histogram | 228–247 |
| 10 | Reduction | 248–277 |
| 11 | Scan | 278–314 |
| 12 | Filter | 315–328 |
| 13 | Merge | 329–354 |
| 14 | Sorting | 355–374 |
| 15 | Advanced optimizations for matrix multiplication | 375–398 |
| 16 | Dynamic programming and wavefront parallelism | 399–426 |
| 17 | Sparse matrix computation | 427–450 |
| 18 | Graph traversal | 451–477 |
| 19 | Convolutional neural networks | 478–500 |
| 20 | Large language models | 501–536 |
| 21 | Electrostatic potential map | 537–552 |
| 22 | Algorithm selection, problem decomposition, problem formulation | 553–564 |
| 23 | Multi-GPU programming | 565–600 |

Extraction command per chapter: `pdftotext -f <start> -l <end> -layout <pdf> -`

## Repo layout

```
pmpp-cuda-samples/
├── README.md
├── common/
│   └── cuda_utils.h
├── part1-fundamental-concepts/
│   ├── ch02-heterogeneous-data-parallel-computing/
│   ├── ch03-multidimensional-grids-and-data/
│   ├── ch04-compute-architecture-and-scheduling/
│   ├── ch05-memory-architecture-and-data-locality/
│   └── ch06-performance-considerations/
├── part2-parallel-patterns/
│   ├── ch07-convolution/
│   ├── ch08-stencil-computation/
│   ├── ch09-histogram/
│   ├── ch10-reduction/
│   ├── ch11-scan/
│   ├── ch12-filter/
│   ├── ch13-merge/
│   ├── ch14-sorting/
│   └── ch15-advanced-matmul-optimizations/
└── part3-advanced-patterns-and-applications/
    ├── ch16-dynamic-programming-and-wavefront-parallelism/
    ├── ch17-sparse-matrix-computation/
    ├── ch18-graph-traversal/
    ├── ch19-convolutional-neural-networks/
    ├── ch20-large-language-models/
    ├── ch21-electrostatic-potential-map/
    ├── ch22-algorithm-selection-and-problem-decomposition/
    └── ch23-multi-gpu-programming/
```

## Per-chapter contents

Each chapter folder gets:

- One `.cu` file per distinct kernel/algorithm variant the chapter
  presents (progressively-optimized versions get separate files, e.g.
  Ch10's naive / divergence-reduced / warp-shuffle reduction kernels).
  File names describe the technique, e.g. `01_naive_reduction.cu`,
  `02_convergent_reduction.cu`, numbered in the order the book introduces
  them.
- Each `.cu` file is self-contained: generates test input, runs a CPU
  reference implementation, runs the CUDA kernel, checks results match
  (tolerance-based for floats), prints pass/fail and timing. `#include
  "../../common/cuda_utils.h"` for the shared `CUDA_CHECK` macro and
  timing helper.
- `Makefile` building every `.cu` in the folder into `bin/<name>`.
- `README.md`: table mapping each file to the book section/listing it
  implements, plus a short paragraph per file on the technique and why it
  improves on the prior version (where applicable).
- Chapter 23 (multi-GPU, MPI/NCCL/NVSHMEM) is a special case: those
  samples require a multi-GPU runtime environment (MPI launcher, NCCL,
  NVSHMEM libraries) that may not be installed. Code is still written and
  compiled if the required library is available; if a library is
  missing, the README notes it and the Makefile target is skipped rather
  than faked.

## Shared utility (`common/cuda_utils.h`)

A single header, header-only, with:
- `CUDA_CHECK(call)` macro — checks the `cudaError_t` return, prints
  file/line and `cudaGetErrorString`, exits on failure.
- A small `CpuTimer`/`GpuTimer` (using `cudaEvent_t`) for consistent
  timing output across samples.
- A `bool nearlyEqual(float a, float b, float eps)` helper for float
  comparisons in correctness checks.

This is the only shared abstraction — everything else stays local to its
chapter folder to keep each sample readable in isolation, matching how the
book presents each chapter's code as self-contained listings.

## Build & verification

- Default target architecture: `-arch=sm_75` (compatible with both local
  GPUs: RTX 2070 SUPER sm_75, RTX 4090 sm_89). Used for every sample
  unless noted below.
- **Exception — `cp.async`/`cuda::memcpy_async` samples**: Ch6.7 (double
  buffering) and Ch15.8 (software pipelining) rely on Ampere+ async-copy
  instructions, which require **sm_80+** and do not run on the 2070
  SUPER (sm_75). Rather than water these down to a manual, non-async
  fallback, those specific `.cu` files are compiled with `-arch=sm_80`
  (own `Makefile` target/variable), implement the real async-copy version
  as the book describes it, and their chapter README notes that they only
  run on sm_80+ GPUs (verified on the 4090 here). Everything else in
  those two chapters still targets sm_75.
- Every `.cu` file is compiled and actually run against its CPU reference
  as part of writing it (async-copy files run on the 4090 specifically).
  A sample is not considered done until it builds clean and its
  correctness check passes on real hardware.
- Top-level `README.md` documents prerequisites (CUDA Toolkit 12.6+,
  `make`, and — for the two sm_80+ samples — a GPU with compute
  capability ≥ 8.0) and how to build all chapters (`make -C
  <chapter-dir>`) or a single chapter.

## Execution plan

Work proceeds chapter-by-chapter (sequential in git history, but chapters
are independent of each other so multiple can be produced in parallel by
subagents). For each chapter:

1. Extract the chapter's page range from the PDF with `pdftotext`.
2. Read the chapter content and identify every distinct
   kernel/algorithm/host-code listing.
3. Write the `.cu` files, `README.md`, and `Makefile` for that chapter.
4. Compile and run each sample; fix until correctness checks pass.
5. Commit the chapter folder to git.

Given the volume (22 chapters), chapters are batched across subagents to
keep the book's raw text out of the orchestrator's context. Each subagent
is self-contained per chapter (or small chapter group) and reports back a
short summary; it does not need prior chapters' content since chapters
don't share code beyond `common/cuda_utils.h`, which is written first.

## Out of scope

- End-of-chapter exercises.
- Appendices A–C.
- GitHub remote creation/push.
- Multi-GPU runtime environments not already installed (Ch 23 degrades
  gracefully per above).
