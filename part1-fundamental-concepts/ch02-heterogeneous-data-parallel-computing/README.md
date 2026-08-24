# Chapter 2: Heterogeneous data-parallel computing

Source: *Programming Massively Parallel Processors*, 5th ed., Ch. 2 (pp. 21-43).

| File | Book section | Technique |
|------|-------------|-----------|
| `01_vector_addition.cu` | §2.3-§2.6 | Sequential host `vecAdd` (Fig. 2.4), `cudaMalloc`/`cudaMemcpy` device data transfer (§2.4, Fig. 2.5/2.8), `vecAddKernel` with the `if (i < n)` boundary guard (§2.5, Fig. 2.10), and the kernel launch with a ceiling-division grid size (§2.6, Fig. 2.12/2.13) |

The chapter's own running example is vector addition, built up in stages across
these sections: the plain sequential C++ version, the device-memory management
that ships data to and from the GPU, the kernel itself, and finally the
`<<<blocksPerGrid, threadsPerBlock>>>` launch. `01_vector_addition.cu` mirrors
that same progression in one file, generates its own input vectors, computes a
CPU reference with the sequential `vecAdd_h`, and compares it against the GPU
result with `nearlyEqual`.

§2.7 (Compilation) and §2.8 (Summary) are prose/conceptual and have no
associated code listing, so no separate file was added for them.

Build and run all samples in this chapter:

```sh
make run
```
