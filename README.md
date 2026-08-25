# PMPP CUDA Samples

![Programming Massively Parallel Processors, 5th Edition cover](Programming_Massively_Parallel_Processors_5E_-_Wen-mei_W_Hwu.jpg)

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

## Chapter index

| Ch | Directory | Topic |
|----|-----------|-------|
| 2 | `part1-fundamental-concepts/ch02-heterogeneous-data-parallel-computing` | Vector addition |
| 3 | `part1-fundamental-concepts/ch03-multidimensional-grids-and-data` | Grayscale, blur, naive matmul |
| 4 | `part1-fundamental-concepts/ch04-compute-architecture-and-scheduling` | Control divergence, device query |
| 5 | `part1-fundamental-concepts/ch05-memory-architecture-and-data-locality` | Tiled matrix multiplication |
| 6 | `part1-fundamental-concepts/ch06-performance-considerations` | Performance-tuning techniques |
| 7 | `part2-parallel-patterns/ch07-convolution` | 1D/2D convolution |
| 8 | `part2-parallel-patterns/ch08-stencil-computation` | Stencil computation |
| 9 | `part2-parallel-patterns/ch09-histogram` | Histogram (atomics, privatization) |
| 10 | `part2-parallel-patterns/ch10-reduction` | Parallel reduction |
| 11 | `part2-parallel-patterns/ch11-scan` | Parallel scan (prefix sum) |
| 12 | `part2-parallel-patterns/ch12-filter` | Stream compaction / filter |
| 13 | `part2-parallel-patterns/ch13-merge` | Parallel merge |
| 14 | `part2-parallel-patterns/ch14-sorting` | Parallel sorting |
| 15 | `part2-parallel-patterns/ch15-advanced-matmul-optimizations` | Advanced matmul optimizations |
| 16 | `part3-advanced-patterns-and-applications/ch16-dynamic-programming-and-wavefront-parallelism` | Dynamic programming, wavefront parallelism |
| 17 | `part3-advanced-patterns-and-applications/ch17-sparse-matrix-computation` | Sparse matrix computation (SpMV) |
| 18 | `part3-advanced-patterns-and-applications/ch18-graph-traversal` | Graph traversal (BFS) |
| 19 | `part3-advanced-patterns-and-applications/ch19-convolutional-neural-networks` | Convolutional neural network layers |
| 20 | `part3-advanced-patterns-and-applications/ch20-large-language-models` | Attention / LLM kernels |
| 21 | `part3-advanced-patterns-and-applications/ch21-electrostatic-potential-map` | Electrostatic potential map |
| 22 | `part3-advanced-patterns-and-applications/ch22-algorithm-selection-and-problem-decomposition` | Algorithm selection, problem decomposition |
| 23 | `part3-advanced-patterns-and-applications/ch23-multi-gpu-programming` | Multi-GPU programming (MPI/NCCL/NVSHMEM) |

## Build & run

Each chapter folder is independent:

```sh
make -C part2-parallel-patterns/ch07-convolution run
```

builds every sample in that chapter and runs it, printing PASS/FAIL and
timing for each.

## Build modes

Every chapter Makefile supports a `DEBUG` toggle. `DEBUG=1` is the
default — `make` / `make run` with no arguments — and builds with
`-O0 -g -G`, producing a binary that is fully steppable with `cuda-gdb`
and correctness-focused rather than fast. Pass `DEBUG=0` (e.g.
`make DEBUG=0 run`) for an optimized, symbol-free `-O2` build. Where a
chapter's own README quotes specific timing numbers, they were measured
under the build mode noted in that chapter's README (most that publish a
timing comparison used `DEBUG=0`, since `-G` skews relative timings) —
check the chapter's README rather than assuming a single repo-wide
convention, and re-measure with `make DEBUG=0 run` if you need to
reproduce a published number yourself.

Chapter 23's `02`-`05` (MPI, MPI+overlap, NCCL, and NVSHMEM multi-GPU
stencil samples, respectively) are written in full but are not buildable
on a machine without the corresponding MPI/NCCL/NVSHMEM development
packages installed — see `part3-advanced-patterns-and-applications/ch23-multi-gpu-programming/README.md`
for details. Every other sample in the repo builds and runs with no
extra dependencies beyond the CUDA Toolkit.

## Scope

Samples implement the kernels, host code, and algorithms the book's
prose and figures actually present, chapter by chapter. End-of-chapter
exercises are not implemented. Appendices A–C and Chapter 1/24 are out
of scope (see the design spec in `docs/superpowers/specs/`).
