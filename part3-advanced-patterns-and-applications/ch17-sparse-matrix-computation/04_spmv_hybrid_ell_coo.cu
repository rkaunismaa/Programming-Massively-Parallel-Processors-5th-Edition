// Chapter 17: Sparse matrix computation
// §17.5: Regulating padding with the hybrid ELL-COO format.
//
// ELL's padding overhead is dominated by whichever row has the most
// non-zeros: every other row must be padded out to that row's length
// (§17.4). §17.5's fix is to cap the ELL part at some row length (`ellCap`)
// and move any non-zeros that overflow a row's cap into a separate COO
// array -- "we can take away some of the elements from the rows with
// exceedingly large number of non-zero elements and place them into a
// separate COO storage. We can use SpMV/ELL on the remaining elements ...
// We can then use a SpMV/COO to finish the job."
//
// This file builds a test matrix with a couple of unusually dense rows
// (mirroring Fig. 17.13's rows 1 and 6) so the regulation actually has an
// effect: capping ELL at `ellCap` keeps most rows unpadded while the
// overflow from the dense rows is handled by SpMV/COO (§17.2, Fig. 17.5)
// atomically accumulating into the same output vector the ELL kernel wrote.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#include "../../common/cuda_utils.h"

// ---------------------------------------------------------------------------
// §17.4, Fig. 17.12: SpMV/ELL part of the hybrid -- one thread per row,
// capped at ellCap non-zeros per row (the overflow lives in COO instead).
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

// ---------------------------------------------------------------------------
// §17.2, Fig. 17.5: SpMV/COO part of the hybrid -- one thread per overflow
// non-zero, atomically accumulating on top of the ELL part's y[row].
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

// Build a dense matrix where most rows have a small number of non-zeros but
// `numOutlierRows` rows are much denser -- the pathological case §17.5
// targets, where a handful of long rows would otherwise force excessive
// ELL padding on every other row.
std::vector<float> generateDenseWithOutliers(int rows, int cols, float baseDensity, float outlierDensity, int numOutlierRows, unsigned int seed) {
    std::vector<float> a(static_cast<size_t>(rows) * cols, 0.0f);
    unsigned int state = seed;
    auto nextRand = [&]() -> float {
        state = state * 1103515245u + 12345u;
        return static_cast<float>((state >> 8) & 0xFFFFFFu) / static_cast<float>(0xFFFFFFu);
    };
    for (int r = 0; r < rows; ++r) {
        float density = (r < numOutlierRows) ? outlierDensity : baseDensity;
        for (int c = 0; c < cols; ++c) {
            if (nextRand() < density) {
                float v = 1.0f + nextRand() * 9.0f;
                a[static_cast<size_t>(r) * cols + c] = v;
            }
        }
    }
    return a;
}

// Convert a dense matrix to the hybrid ELL-COO format: each row keeps up to
// `ellCap` non-zeros in the (padded, column-major) ELL part; anything past
// that spills into the COO arrays.
void denseToHybridEllCoo(const std::vector<float> &a, int rows, int cols, int ellCap,
                          std::vector<int> &ellColIdx, std::vector<float> &ellValue, std::vector<int> &ellNnzPerRow,
                          std::vector<int> &cooRowIdx, std::vector<int> &cooColIdx, std::vector<float> &cooValue) {
    ellColIdx.assign(static_cast<size_t>(ellCap) * rows, 0);
    ellValue.assign(static_cast<size_t>(ellCap) * rows, 0.0f);
    ellNnzPerRow.assign(rows, 0);
    cooRowIdx.clear();
    cooColIdx.clear();
    cooValue.clear();

    for (int r = 0; r < rows; ++r) {
        int t = 0;
        for (int c = 0; c < cols; ++c) {
            float v = a[static_cast<size_t>(r) * cols + c];
            if (v == 0.0f) continue;
            if (t < ellCap) {
                int i = t * rows + r;
                ellColIdx[i] = c;
                ellValue[i] = v;
                ++t;
            } else {
                cooRowIdx.push_back(r);
                cooColIdx.push_back(c);
                cooValue.push_back(v);
            }
        }
        ellNnzPerRow[r] = t;
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

float runSpmvHybrid(const std::vector<int> &ellColIdx_h, const std::vector<float> &ellValue_h, const std::vector<int> &ellNnzPerRow_h,
                     const std::vector<int> &cooRowIdx_h, const std::vector<int> &cooColIdx_h, const std::vector<float> &cooValue_h,
                     int rows, const std::vector<float> &x_h, std::vector<float> &y_h) {
    int ellSize = static_cast<int>(ellValue_h.size());
    int cooNnz = static_cast<int>(cooValue_h.size());
    int cols = static_cast<int>(x_h.size());

    int *ellColIdx_d, *ellNnzPerRow_d, *cooRowIdx_d, *cooColIdx_d;
    float *ellValue_d, *cooValue_d, *x_d, *y_d;
    CUDA_CHECK(cudaMalloc(&ellColIdx_d, ellSize * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ellValue_d, ellSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ellNnzPerRow_d, rows * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&x_d, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&y_d, rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cooRowIdx_d, std::max(cooNnz, 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cooColIdx_d, std::max(cooNnz, 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&cooValue_d, std::max(cooNnz, 1) * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(ellColIdx_d, ellColIdx_h.data(), ellSize * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ellValue_d, ellValue_h.data(), ellSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ellNnzPerRow_d, ellNnzPerRow_h.data(), rows * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(x_d, x_h.data(), cols * sizeof(float), cudaMemcpyHostToDevice));
    if (cooNnz > 0) {
        CUDA_CHECK(cudaMemcpy(cooRowIdx_d, cooRowIdx_h.data(), cooNnz * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(cooColIdx_d, cooColIdx_h.data(), cooNnz * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(cooValue_d, cooValue_h.data(), cooNnz * sizeof(float), cudaMemcpyHostToDevice));
    }

    int blockDim = 256;
    int ellGridDim = (rows + blockDim - 1) / blockDim;
    int cooGridDim = (cooNnz + blockDim - 1) / blockDim;

    auto launch = [&]() {
        // ELL part assigns (writes) y[row] first; COO part then atomically
        // adds the overflow contributions on top -- together they compute
        // the full dense matvec.
        spmvEllKernel<<<ellGridDim, blockDim>>>(ellColIdx_d, ellValue_d, ellNnzPerRow_d, rows, x_d, y_d);
        CUDA_CHECK(cudaGetLastError());
        if (cooNnz > 0) {
            spmvCooKernel<<<cooGridDim, blockDim>>>(cooRowIdx_d, cooColIdx_d, cooValue_d, cooNnz, x_d, y_d);
            CUDA_CHECK(cudaGetLastError());
        }
    };

    launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    launch();
    float ms = timer.stopAndGetMs();

    y_h.resize(rows);
    CUDA_CHECK(cudaMemcpy(y_h.data(), y_d, rows * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(ellColIdx_d));
    CUDA_CHECK(cudaFree(ellValue_d));
    CUDA_CHECK(cudaFree(ellNnzPerRow_d));
    CUDA_CHECK(cudaFree(x_d));
    CUDA_CHECK(cudaFree(y_d));
    CUDA_CHECK(cudaFree(cooRowIdx_d));
    CUDA_CHECK(cudaFree(cooColIdx_d));
    CUDA_CHECK(cudaFree(cooValue_d));
    return ms;
}

bool checkClose(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (!nearlyEqual(a[i], b[i])) return false;
    return true;
}

bool runCase(int rows, int cols, float baseDensity, float outlierDensity, int numOutlierRows, int ellCap, unsigned int seed) {
    std::vector<float> a = generateDenseWithOutliers(rows, cols, baseDensity, outlierDensity, numOutlierRows, seed);

    std::vector<int> ellColIdx, ellNnzPerRow, cooRowIdx, cooColIdx;
    std::vector<float> ellValue, cooValue;
    denseToHybridEllCoo(a, rows, cols, ellCap, ellColIdx, ellValue, ellNnzPerRow, cooRowIdx, cooColIdx, cooValue);

    // A plain (uncapped) ELL representation would need to pad every row out
    // to the true max row length; report how many elements that saves.
    int trueMaxNnzPerRow = 0;
    for (int r = 0; r < rows; ++r) {
        int cnt = 0;
        for (int c = 0; c < cols; ++c)
            if (a[static_cast<size_t>(r) * cols + c] != 0.0f) ++cnt;
        trueMaxNnzPerRow = std::max(trueMaxNnzPerRow, cnt);
    }
    long long uncappedEllElems = static_cast<long long>(trueMaxNnzPerRow) * rows;
    long long hybridElems = static_cast<long long>(ellCap) * rows + static_cast<long long>(cooValue.size());

    std::vector<float> x(cols);
    unsigned int state = seed ^ 0xABCDu;
    for (int c = 0; c < cols; ++c) {
        state = state * 1103515245u + 12345u;
        x[c] = static_cast<float>((state >> 8) & 0xFFu) / 25.0f;
    }
    std::vector<float> ref = cpuMatVec(a, rows, cols, x);

    std::vector<float> y;
    float ms = runSpmvHybrid(ellColIdx, ellValue, ellNnzPerRow, cooRowIdx, cooColIdx, cooValue, rows, x, y);

    bool ok = checkClose(y, ref);
    printf("%dx%d ellCap=%d trueMaxRow=%d cooOverflow=%zu (padded elems %lld vs uncapped ELL %lld): %.4f ms  [%s]\n",
           rows, cols, ellCap, trueMaxNnzPerRow, cooValue.size(), hybridElems, uncappedEllElems, ms, ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    printf("SpMV with hybrid ELL-COO format: capped ELL + COO overflow for outlier rows (§17.5):\n");
    bool ok = true;
    // A handful of rows are much denser than the rest, forcing an uncapped
    // ELL representation into heavy padding -- exactly the case §17.5
    // targets. ellCap=4 keeps most (sparse) rows unpadded.
    ok = runCase(400, 300, 0.02f, 0.6f, 3, 4, 11u) && ok;
    ok = runCase(64, 64, 0.05f, 0.8f, 2, 3, 99u) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
