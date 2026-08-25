// Chapter 17: Sparse matrix computation
// §17.7, Fig. 17.18: Column-wise accessibility with the CSC format.
//
// Compressed Sparse Column (CSC) mirrors CSR with rows and columns swapped:
// `value`/`rowIdx` group non-zeros by COLUMN, and `colPtrs` (size
// numCols+1) gives each column's starting offset. §17.7 is explicit that
// "CSC is not intended to be used for performing SpMV" but walks through an
// SpMV/CSC kernel anyway "for completeness, and for the interesting
// exercise of comparing its advantages and disadvantages to other formats"
// -- Fig. 17.17/17.18 -- so that is exactly what this file implements.
//
// Parallelization: one thread per COLUMN. Each thread loads x[col] ONCE
// (coalesced across threads -- CSC's one genuine advantage for SpMV, per
// §17.7), then walks that column's non-zeros (colPtrs[col]..colPtrs[col+1]),
// multiplying by the cached x[col] and ATOMICALLY accumulating into y[row]
// -- an atomic is required because different columns (different threads)
// can both have a non-zero in the same row.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

// ---------------------------------------------------------------------------
// §17.7, Fig. 17.18: SpMV/CSC -- one thread per column, atomic accumulate
// into the output vector (coalesced read of the input vector x).
// ---------------------------------------------------------------------------
__global__ void spmvCscKernel(const int *rowIdx, const float *value, const int *colPtrs, int numCols, const float *x, float *y) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < numCols) {
        float xVal = x[col];
        for (int i = colPtrs[col]; i < colPtrs[col + 1]; ++i) {
            int row = rowIdx[i];
            float val = value[i];
            atomicAdd(&y[row], val * xVal);
        }
    }
}

std::vector<float> generateDenseSparse(int rows, int cols, float density, unsigned int seed) {
    std::vector<float> a(static_cast<size_t>(rows) * cols, 0.0f);
    unsigned int state = seed;
    auto nextRand = [&]() -> float {
        state = state * 1103515245u + 12345u;
        return static_cast<float>((state >> 8) & 0xFFFFFFu) / static_cast<float>(0xFFFFFFu);
    };
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (nextRand() < density) {
                float v = 1.0f + nextRand() * 9.0f;
                a[static_cast<size_t>(r) * cols + c] = v;
            }
        }
    }
    return a;
}

// Convert a dense matrix to CSC: non-zeros grouped by column, `colPtrs`
// giving each column's starting offset (Fig. 17.16).
void denseToCsc(const std::vector<float> &a, int rows, int cols, std::vector<int> &colPtrs, std::vector<int> &rowIdx, std::vector<float> &value) {
    colPtrs.assign(cols + 1, 0);
    rowIdx.clear();
    value.clear();
    for (int c = 0; c < cols; ++c) {
        colPtrs[c] = static_cast<int>(value.size());
        for (int r = 0; r < rows; ++r) {
            float v = a[static_cast<size_t>(r) * cols + c];
            if (v != 0.0f) {
                rowIdx.push_back(r);
                value.push_back(v);
            }
        }
    }
    colPtrs[cols] = static_cast<int>(value.size());
}

std::vector<float> cpuMatVec(const std::vector<float> &a, int rows, int cols, const std::vector<float> &x) {
    std::vector<float> y(rows, 0.0f);
    for (int r = 0; r < rows; ++r) {
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) sum += a[static_cast<size_t>(r) * cols + c] * x[c];
        y[r] = sum;
    }
    return y;
}

float runSpmvCsc(const std::vector<int> &colPtrs_h, const std::vector<int> &rowIdx_h, const std::vector<float> &value_h, int rows, const std::vector<float> &x_h, std::vector<float> &y_h) {
    int nnz = static_cast<int>(value_h.size());
    int cols = static_cast<int>(x_h.size());

    int *colPtrs_d, *rowIdx_d;
    float *value_d, *x_d, *y_d;
    CUDA_CHECK(cudaMalloc(&colPtrs_d, (cols + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&rowIdx_d, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&value_d, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&x_d, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&y_d, rows * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(colPtrs_d, colPtrs_h.data(), (cols + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(rowIdx_d, rowIdx_h.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(value_d, value_h.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(x_d, x_h.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

    int blockDim = 256;
    int gridDim = (cols + blockDim - 1) / blockDim;

    auto launch = [&]() {
        CUDA_CHECK(cudaMemset(y_d, 0, rows * sizeof(float)));
        spmvCscKernel<<<gridDim, blockDim>>>(rowIdx_d, value_d, colPtrs_d, cols, x_d, y_d);
        CUDA_CHECK(cudaGetLastError());
    };

    launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    launch();
    float ms = timer.stopAndGetMs();

    y_h.resize(rows);
    CUDA_CHECK(cudaMemcpy(y_h.data(), y_d, rows * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(colPtrs_d));
    CUDA_CHECK(cudaFree(rowIdx_d));
    CUDA_CHECK(cudaFree(value_d));
    CUDA_CHECK(cudaFree(x_d));
    CUDA_CHECK(cudaFree(y_d));
    return ms;
}

bool checkClose(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (!nearlyEqual(a[i], b[i])) return false;
    return true;
}

// Same canonical 4x4 matrix used throughout the chapter; colPtrs should come
// out as [0, 2, 4, 6, 8] per Fig. 17.16 (every column has exactly 2 non-zeros
// in this particular example).
bool runCanonicalCase() {
    const int rows = 4, cols = 4;
    std::vector<float> a = {
        1, 7, 0, 0,
        5, 0, 3, 9,
        0, 2, 8, 0,
        0, 0, 0, 6};
    std::vector<int> colPtrs, rowIdx;
    std::vector<float> value;
    denseToCsc(a, rows, cols, colPtrs, rowIdx, value);

    bool colPtrsOk = (colPtrs == std::vector<int>{0, 2, 4, 6, 8});

    std::vector<float> x = {1, 1, 1, 1};
    std::vector<float> ref = cpuMatVec(a, rows, cols, x);

    std::vector<float> y;
    float ms = runSpmvCsc(colPtrs, rowIdx, value, rows, x, y);

    bool ok = checkClose(y, ref) && colPtrsOk;
    printf("canonical 4x4 (nnz=%zu, colPtrs %s): %.4f ms  [%s]\n", value.size(), colPtrsOk ? "as expected" : "UNEXPECTED", ms, ok ? "match" : "MISMATCH");
    return ok;
}

bool runRandomCase(int rows, int cols, float density, unsigned int seed) {
    std::vector<float> a = generateDenseSparse(rows, cols, density, seed);
    std::vector<int> colPtrs, rowIdx;
    std::vector<float> value;
    denseToCsc(a, rows, cols, colPtrs, rowIdx, value);

    std::vector<float> x(cols);
    unsigned int state = seed ^ 0xABCDu;
    for (int c = 0; c < cols; ++c) {
        state = state * 1103515245u + 12345u;
        x[c] = static_cast<float>((state >> 8) & 0xFFu) / 25.0f;
    }
    std::vector<float> ref = cpuMatVec(a, rows, cols, x);

    std::vector<float> y;
    float ms = runSpmvCsc(colPtrs, rowIdx, value, rows, x, y);

    bool ok = checkClose(y, ref);
    printf("random %dx%d density=%.2f (nnz=%zu): %.4f ms  [%s]\n", rows, cols, density, value.size(), ms, ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    printf("SpMV with CSC format, one thread per column, atomic accumulate (§17.7, Fig. 17.18):\n");
    bool ok = true;
    ok = runCanonicalCase() && ok;
    ok = runRandomCase(500, 400, 0.05f, 42u) && ok;
    ok = runRandomCase(1024, 1024, 0.01f, 7u) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
