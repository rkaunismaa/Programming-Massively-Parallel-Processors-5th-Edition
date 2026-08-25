// Chapter 17: Sparse matrix computation
// §17.2, Fig. 17.5: A simple SpMV kernel with the COO format.
//
// The Coordinate (COO) format stores every non-zero as a triple
// (rowIdx[i], colIdx[i], value[i]) in three parallel one-dimensional arrays
// -- no assumption about ordering is required for correctness (§17.2: "The
// SpMV kernel presented in the section that uses the COO format works even
// if the non-zeros are not sorted").
//
// Parallelization (Fig. 17.4/17.5): one thread per non-zero element. Each
// thread looks up its row/col/value, multiplies value by x[col], and
// ATOMICALLY accumulates into y[row] -- an atomic is required because
// multiple non-zeros of the same row (handled by different threads) update
// the same output element, and there is no cheap way to statically avoid
// the collision in the COO layout.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

// ---------------------------------------------------------------------------
// §17.2, Fig. 17.5: SpMV/COO -- one thread per non-zero, atomic accumulate.
// ---------------------------------------------------------------------------
__global__ void spmvCooKernel(const int *rowIdx, const int *colIdx, const float *value, int nnz, const float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nnz) {
        int row = rowIdx[i];
        int col = colIdx[i];
        float val = value[i];
        atomicAdd(&y[row], val * x[col]);
    }
}

// A dense matrix stored row-major, generated with a controllable density of
// non-zero entries. Every file in this chapter builds its own small sparse
// test matrix, converts it to the format under test on the host, and checks
// the GPU SpMV result against a dense CPU reference matvec.
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
                float v = 1.0f + nextRand() * 9.0f;  // non-zero in [1, 10)
                a[static_cast<size_t>(r) * cols + c] = v;
            }
        }
    }
    return a;
}

void denseToCoo(const std::vector<float> &a, int rows, int cols, std::vector<int> &rowIdx, std::vector<int> &colIdx, std::vector<float> &value) {
    rowIdx.clear();
    colIdx.clear();
    value.clear();
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float v = a[static_cast<size_t>(r) * cols + c];
            if (v != 0.0f) {
                rowIdx.push_back(r);
                colIdx.push_back(c);
                value.push_back(v);
            }
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

float runSpmvCoo(const std::vector<int> &rowIdx_h, const std::vector<int> &colIdx_h, const std::vector<float> &value_h, int rows, const std::vector<float> &x_h, std::vector<float> &y_h) {
    int nnz = static_cast<int>(value_h.size());
    int cols = static_cast<int>(x_h.size());

    int *rowIdx_d, *colIdx_d;
    float *value_d, *x_d, *y_d;
    CUDA_CHECK(cudaMalloc(&rowIdx_d, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&colIdx_d, nnz * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&value_d, nnz * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&x_d, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&y_d, rows * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(rowIdx_d, rowIdx_h.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(colIdx_d, colIdx_h.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(value_d, value_h.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(x_d, x_h.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

    int blockDim = 256;
    int gridDim = (nnz + blockDim - 1) / blockDim;

    auto launch = [&]() {
        CUDA_CHECK(cudaMemset(y_d, 0, rows * sizeof(float)));
        spmvCooKernel<<<gridDim, blockDim>>>(rowIdx_d, colIdx_d, value_d, nnz, x_d, y_d);
        CUDA_CHECK(cudaGetLastError());
    };

    launch();  // warm-up
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    launch();
    float ms = timer.stopAndGetMs();

    y_h.resize(rows);
    CUDA_CHECK(cudaMemcpy(y_h.data(), y_d, rows * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(rowIdx_d));
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

// §17.2's own worked example: a 4x4 matrix used throughout the chapter
// (Figs. 17.3, 17.7, 17.10, 17.16), reconstructed from the row/column
// groupings the chapter text gives for CSR (Fig. 17.7) and CSC (Fig. 17.16):
//   [1 7 0 0]
//   [5 0 3 9]
//   [0 2 8 0]
//   [0 0 0 6]
bool runCanonicalCase() {
    const int rows = 4, cols = 4;
    std::vector<float> a = {
        1, 7, 0, 0,
        5, 0, 3, 9,
        0, 2, 8, 0,
        0, 0, 0, 6};
    std::vector<int> rowIdx, colIdx;
    std::vector<float> value;
    denseToCoo(a, rows, cols, rowIdx, colIdx, value);

    std::vector<float> x = {1, 1, 1, 1};
    std::vector<float> ref = cpuMatVec(a, rows, cols, x);

    std::vector<float> y;
    float ms = runSpmvCoo(rowIdx, colIdx, value, rows, x, y);

    bool ok = checkClose(y, ref);
    printf("canonical 4x4 (nnz=%zu): %.4f ms  [%s]\n", value.size(), ms, ok ? "match" : "MISMATCH");
    return ok;
}

bool runRandomCase(int rows, int cols, float density, unsigned int seed) {
    std::vector<float> a = generateDenseSparse(rows, cols, density, seed);
    std::vector<int> rowIdx, colIdx;
    std::vector<float> value;
    denseToCoo(a, rows, cols, rowIdx, colIdx, value);

    std::vector<float> x(cols);
    unsigned int state = seed ^ 0xABCDu;
    for (int c = 0; c < cols; ++c) {
        state = state * 1103515245u + 12345u;
        x[c] = static_cast<float>((state >> 8) & 0xFFu) / 25.0f;
    }
    std::vector<float> ref = cpuMatVec(a, rows, cols, x);

    std::vector<float> y;
    float ms = runSpmvCoo(rowIdx, colIdx, value, rows, x, y);

    bool ok = checkClose(y, ref);
    printf("random %dx%d density=%.2f (nnz=%zu): %.4f ms  [%s]\n", rows, cols, density, value.size(), ms, ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    printf("SpMV with COO format, one thread per non-zero, atomic accumulate (§17.2, Fig. 17.5):\n");
    bool ok = true;
    ok = runCanonicalCase() && ok;
    ok = runRandomCase(500, 400, 0.05f, 42u) && ok;
    ok = runRandomCase(1024, 1024, 0.01f, 7u) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
