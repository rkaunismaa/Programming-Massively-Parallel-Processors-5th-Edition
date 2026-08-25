// Chapter 17: Sparse matrix computation
// §17.3, Fig. 17.9: Grouping row non-zeros with the CSR format.
//
// The Compressed Sparse Row (CSR) format groups non-zeros by row: `value`
// and `colIdx` store the non-zeros row-by-row, and a `rowPtrs` array (size
// numRows+1) gives the starting offset of each row's non-zeros -- rowPtrs[r]
// is where row r begins and rowPtrs[r+1] is where it ends (with rowPtrs[R]
// marking the end of the last row).
//
// Parallelization (Fig. 17.8/17.9): one thread per ROW. Each thread walks
// its own row's non-zeros (rowPtrs[row]..rowPtrs[row+1]-1), accumulates a
// dot product with x into a private `sum`, and writes sum to y[row] once at
// the end -- since each row is owned by exactly one thread, NO atomics are
// needed (unlike SpMV/COO). The tradeoff (§17.3): consecutive threads read
// far-apart locations in `value`/`colIdx` on each loop iteration, so this
// kernel's matrix accesses are not coalesced, and rows with very different
// lengths cause control divergence within a warp.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

// ---------------------------------------------------------------------------
// §17.3, Fig. 17.9: SpMV/CSR -- one thread per row, no atomics needed.
// ---------------------------------------------------------------------------
__global__ void spmvCsrKernel(const int *rowPtrs, const int *colIdx, const float *value, int numRows, const float *x, float *y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows) {
        float sum = 0.0f;
        for (int i = rowPtrs[row]; i < rowPtrs[row + 1]; ++i) {
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

void denseToCsr(const std::vector<float> &a, int rows, int cols, std::vector<int> &rowPtrs, std::vector<int> &colIdx, std::vector<float> &value) {
    rowPtrs.assign(rows + 1, 0);
    colIdx.clear();
    value.clear();
    for (int r = 0; r < rows; ++r) {
        rowPtrs[r] = static_cast<int>(value.size());
        for (int c = 0; c < cols; ++c) {
            float v = a[static_cast<size_t>(r) * cols + c];
            if (v != 0.0f) {
                colIdx.push_back(c);
                value.push_back(v);
            }
        }
    }
    rowPtrs[rows] = static_cast<int>(value.size());
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

float runSpmvCsr(const std::vector<int> &rowPtrs_h, const std::vector<int> &colIdx_h, const std::vector<float> &value_h, int rows, const std::vector<float> &x_h, std::vector<float> &y_h) {
    int nnz = static_cast<int>(value_h.size());
    int cols = static_cast<int>(x_h.size());

    int *rowPtrs_d, *colIdx_d;
    float *value_d, *x_d, *y_d;
    CUDA_CHECK(cudaMalloc(&rowPtrs_d, (rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&colIdx_d, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&value_d, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&x_d, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&y_d, rows * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(rowPtrs_d, rowPtrs_h.data(), (rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(colIdx_d, colIdx_h.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(value_d, value_h.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(x_d, x_h.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

    int blockDim = 256;
    int gridDim = (rows + blockDim - 1) / blockDim;

    auto launch = [&]() {
        spmvCsrKernel<<<gridDim, blockDim>>>(rowPtrs_d, colIdx_d, value_d, rows, x_d, y_d);
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

    CUDA_CHECK(cudaFree(rowPtrs_d));
    CUDA_CHECK(cudaFree(colIdx_d));
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

// Same canonical 4x4 matrix used throughout the chapter (Figs. 17.3, 17.7,
// 17.10, 17.16); rowPtrs should come out as [0, 2, 5, 7, 8] per Fig. 17.7.
bool runCanonicalCase() {
    const int rows = 4, cols = 4;
    std::vector<float> a = {
        1, 7, 0, 0,
        5, 0, 3, 9,
        0, 2, 8, 0,
        0, 0, 0, 6};
    std::vector<int> rowPtrs, colIdx;
    std::vector<float> value;
    denseToCsr(a, rows, cols, rowPtrs, colIdx, value);

    bool rowPtrsOk = (rowPtrs == std::vector<int>{0, 2, 5, 7, 8});

    std::vector<float> x = {1, 1, 1, 1};
    std::vector<float> ref = cpuMatVec(a, rows, cols, x);

    std::vector<float> y;
    float ms = runSpmvCsr(rowPtrs, colIdx, value, rows, x, y);

    bool ok = checkClose(y, ref) && rowPtrsOk;
    printf("canonical 4x4 (nnz=%zu, rowPtrs %s): %.4f ms  [%s]\n", value.size(), rowPtrsOk ? "as expected" : "UNEXPECTED", ms, ok ? "match" : "MISMATCH");
    return ok;
}

bool runRandomCase(int rows, int cols, float density, unsigned int seed) {
    std::vector<float> a = generateDenseSparse(rows, cols, density, seed);
    std::vector<int> rowPtrs, colIdx;
    std::vector<float> value;
    denseToCsr(a, rows, cols, rowPtrs, colIdx, value);

    std::vector<float> x(cols);
    unsigned int state = seed ^ 0xABCDu;
    for (int c = 0; c < cols; ++c) {
        state = state * 1103515245u + 12345u;
        x[c] = static_cast<float>((state >> 8) & 0xFFu) / 25.0f;
    }
    std::vector<float> ref = cpuMatVec(a, rows, cols, x);

    std::vector<float> y;
    float ms = runSpmvCsr(rowPtrs, colIdx, value, rows, x, y);

    bool ok = checkClose(y, ref);
    printf("random %dx%d density=%.2f (nnz=%zu): %.4f ms  [%s]\n", rows, cols, density, value.size(), ms, ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    printf("SpMV with CSR format, one thread per row, no atomics (§17.3, Fig. 17.9):\n");
    bool ok = true;
    ok = runCanonicalCase() && ok;
    ok = runRandomCase(500, 400, 0.05f, 42u) && ok;
    ok = runRandomCase(1024, 1024, 0.01f, 7u) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
