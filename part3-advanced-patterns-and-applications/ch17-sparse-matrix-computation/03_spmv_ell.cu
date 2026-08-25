// Chapter 17: Sparse matrix computation
// §17.4, Fig. 17.12: Improving memory coalescing with the ELL format.
//
// ELL starts from a CSR-like row grouping, pads every row with zero
// elements up to the length of the longest row (making the matrix
// rectangular), then lays the padded matrix out in COLUMN-MAJOR order
// (§17.4: "equivalent to transposing the rectangular matrix in the row
// major order used by the C language"). Element t of row `row` lives at
// flat index `i = t*numRows + row`.
//
// Parallelization (Fig. 17.11/17.12): one thread per row, same as CSR, but
// now consecutive threads (consecutive `row`) read consecutive addresses on
// every iteration `t` since `i = t*numRows + row` -- so SpMV/ELL's matrix
// accesses ARE coalesced, unlike SpMV/CSR. Per the book, the kernel is
// driven by an `nnzPerRow` array so each thread stops at its row's actual
// non-zero count rather than looping over padding (Fig. 17.12, line 05).

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#include "../../common/cuda_utils.h"

// ---------------------------------------------------------------------------
// §17.4, Fig. 17.12: SpMV/ELL -- one thread per row, column-major storage
// gives coalesced accesses to colIdx/value.
// ---------------------------------------------------------------------------
__global__ void spmvEllKernel(const int *colIdx, const float *value, const int *nnzPerRow, int numRows, const float *x, float *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows) {
        float sum = 0.0f;
        int rowLen = nnzPerRow[row];
        for (int t = 0; t < rowLen; ++t) {
            int i = t * numRows + row;
            int col = colIdx[i];
            float val = value[i];
            sum += val * x[col];
        }
        y[row] = sum;
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

// Convert a dense matrix to ELL: pad every row to `maxNnzPerRow` (the
// longest row), store column-major (Fig. 17.10).
void denseToEll(const std::vector<float> &a, int rows, int cols, std::vector<int> &colIdx, std::vector<float> &value, std::vector<int> &nnzPerRow, int &maxNnzPerRow) {
    std::vector<std::vector<int>> rowCols(rows);
    std::vector<std::vector<float>> rowVals(rows);
    nnzPerRow.assign(rows, 0);
    maxNnzPerRow = 0;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float v = a[static_cast<size_t>(r) * cols + c];
            if (v != 0.0f) {
                rowCols[r].push_back(c);
                rowVals[r].push_back(v);
            }
        }
        nnzPerRow[r] = static_cast<int>(rowCols[r].size());
        maxNnzPerRow = std::max(maxNnzPerRow, nnzPerRow[r]);
    }

    colIdx.assign(static_cast<size_t>(maxNnzPerRow) * rows, 0);
    value.assign(static_cast<size_t>(maxNnzPerRow) * rows, 0.0f);
    for (int r = 0; r < rows; ++r) {
        for (int t = 0; t < nnzPerRow[r]; ++t) {
            int i = t * rows + r;
            colIdx[i] = rowCols[r][t];
            value[i] = rowVals[r][t];
        }
    }
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

float runSpmvEll(const std::vector<int> &colIdx_h, const std::vector<float> &value_h, const std::vector<int> &nnzPerRow_h, int rows, const std::vector<float> &x_h, std::vector<float> &y_h) {
    int ellSize = static_cast<int>(value_h.size());
    int cols = static_cast<int>(x_h.size());

    int *colIdx_d, *nnzPerRow_d;
    float *value_d, *x_d, *y_d;
    CUDA_CHECK(cudaMalloc(&colIdx_d, ellSize * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&value_d, ellSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&nnzPerRow_d, rows * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&x_d, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&y_d, rows * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(colIdx_d, colIdx_h.data(), ellSize * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(value_d, value_h.data(), ellSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(nnzPerRow_d, nnzPerRow_h.data(), rows * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(x_d, x_h.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

    int blockDim = 256;
    int gridDim = (rows + blockDim - 1) / blockDim;

    auto launch = [&]() {
        spmvEllKernel<<<gridDim, blockDim>>>(colIdx_d, value_d, nnzPerRow_d, rows, x_d, y_d);
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

    CUDA_CHECK(cudaFree(colIdx_d));
    CUDA_CHECK(cudaFree(value_d));
    CUDA_CHECK(cudaFree(nnzPerRow_d));
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

// Same canonical 4x4 matrix used throughout the chapter. §17.4's own worked
// example on this matrix pads row lengths [2,3,2,1] up to maxNnzPerRow=3
// (row 1 is the longest), which we assert here.
bool runCanonicalCase() {
    const int rows = 4, cols = 4;
    std::vector<float> a = {
        1, 7, 0, 0,
        5, 0, 3, 9,
        0, 2, 8, 0,
        0, 0, 0, 6};
    std::vector<int> colIdx, nnzPerRow;
    std::vector<float> value;
    int maxNnzPerRow = 0;
    denseToEll(a, rows, cols, colIdx, value, nnzPerRow, maxNnzPerRow);

    bool shapeOk = (maxNnzPerRow == 3) && (nnzPerRow == std::vector<int>{2, 3, 2, 1});

    std::vector<float> x = {1, 1, 1, 1};
    std::vector<float> ref = cpuMatVec(a, rows, cols, x);

    std::vector<float> y;
    float ms = runSpmvEll(colIdx, value, nnzPerRow, rows, x, y);

    bool ok = checkClose(y, ref) && shapeOk;
    printf("canonical 4x4 (maxNnzPerRow=%d, %s): %.4f ms  [%s]\n", maxNnzPerRow, shapeOk ? "as expected" : "UNEXPECTED", ms, ok ? "match" : "MISMATCH");
    return ok;
}

bool runRandomCase(int rows, int cols, float density, unsigned int seed) {
    std::vector<float> a = generateDenseSparse(rows, cols, density, seed);
    std::vector<int> colIdx, nnzPerRow;
    std::vector<float> value;
    int maxNnzPerRow = 0;
    denseToEll(a, rows, cols, colIdx, value, nnzPerRow, maxNnzPerRow);

    std::vector<float> x(cols);
    unsigned int state = seed ^ 0xABCDu;
    for (int c = 0; c < cols; ++c) {
        state = state * 1103515245u + 12345u;
        x[c] = static_cast<float>((state >> 8) & 0xFFu) / 25.0f;
    }
    std::vector<float> ref = cpuMatVec(a, rows, cols, x);

    std::vector<float> y;
    float ms = runSpmvEll(colIdx, value, nnzPerRow, rows, x, y);

    bool ok = checkClose(y, ref);
    printf("random %dx%d density=%.2f (maxNnzPerRow=%d): %.4f ms  [%s]\n", rows, cols, density, maxNnzPerRow, ms, ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    printf("SpMV with ELL format, one thread per row, column-major padded storage (§17.4, Fig. 17.12):\n");
    bool ok = true;
    ok = runCanonicalCase() && ok;
    ok = runRandomCase(500, 400, 0.05f, 42u) && ok;
    ok = runRandomCase(1024, 1024, 0.01f, 7u) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
