// Chapter 17: Sparse matrix computation
// §17.6, Fig. 17.14/17.15: Reducing control divergence with the JDS format.
//
// Jagged Diagonal Storage (JDS) sorts the matrix's rows by length -- "from
// the longest to the shortest" (§17.6) -- keeping a `rows` array that
// records each sorted position's original row index, so the final answer
// can be permuted back afterward. The non-zeros of the sorted rows are then
// stored column-major like ELL, but WITHOUT padding: since row lengths only
// decrease as we go down the sorted order, iteration t's column is exactly
// the (shrinking) prefix of rows whose length is > t. An `iterPtr` array
// (size maxRowLen+1) records where each iteration's column begins in the
// flattened `value`/`colIdx` arrays.
//
// For a sorted row at position r with length L, its t-th non-zero
// (0 <= t < L) sits at flat index `iterPtr[t] + r` -- since at iteration t
// only the r' < count_t longest rows still participate, and this row (being
// among the longest count_t rows, because L > t) is at local offset r
// within that iteration's block.
//
// §17.6's text describes this construction and Fig. 17.15's physical view
// in full ("the threads access the non-zeros and column indices in the JDS
// arrays in a coalesced manner") but states the SpMV/JDS kernel code itself
// "is left as an exercise" without printing a figure for it -- so this file
// implements the one-thread-per-(sorted)-row kernel the prose above
// describes directly, the same treatment this project's ch14 gave §14.3/
// §14.6/§14.8's identically-phrased "left as an exercise" passages that
// come with a full walkthrough.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <numeric>

#include "../../common/cuda_utils.h"

// ---------------------------------------------------------------------------
// §17.6: SpMV/JDS -- one thread per (length-sorted) row; iterPtr locates
// each iteration's shrinking column in the unpadded, column-major storage.
// ---------------------------------------------------------------------------
__global__ void spmvJdsKernel(const int *colIdx, const float *value, const int *rowNnz, const int *iterPtr, int numRows, const float *x, float *ySorted) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < numRows) {
        float sum = 0.0f;
        int len = rowNnz[r];
        for (int t = 0; t < len; ++t) {
            int idx = iterPtr[t] + r;
            int col = colIdx[idx];
            float val = value[idx];
            sum += val * x[col];
        }
        ySorted[r] = sum;
    }
}

// Build a dense matrix whose rows have deliberately varied non-zero counts
// (some long, some short) so sorting by length actually reorders rows --
// the case JDS targets for reducing control divergence.
std::vector<float> generateDenseVariedRowLengths(int rows, int cols, unsigned int seed) {
    std::vector<float> a(static_cast<size_t>(rows) * cols, 0.0f);
    unsigned int state = seed;
    auto nextRand = [&]() -> float {
        state = state * 1103515245u + 12345u;
        return static_cast<float>((state >> 8) & 0xFFFFFFu) / static_cast<float>(0xFFFFFFu);
    };
    for (int r = 0; r < rows; ++r) {
        // Row density cycles across a wide range so row lengths vary a lot.
        float density = 0.02f + 0.5f * (static_cast<float>(r % 7) / 6.0f);
        for (int c = 0; c < cols; ++c) {
            if (nextRand() < density) {
                float v = 1.0f + nextRand() * 9.0f;
                a[static_cast<size_t>(r) * cols + c] = v;
            }
        }
    }
    return a;
}

// Convert a dense matrix to JDS: rows sorted by descending non-zero count,
// unpadded column-major storage with an iterPtr per iteration.
void denseToJds(const std::vector<float> &a, int rows, int cols,
                 std::vector<int> &colIdx, std::vector<float> &value, std::vector<int> &rowNnzSorted,
                 std::vector<int> &iterPtr, std::vector<int> &origRow, int &maxRowLen) {
    std::vector<std::vector<int>> rowCols(rows);
    std::vector<std::vector<float>> rowVals(rows);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float v = a[static_cast<size_t>(r) * cols + c];
            if (v != 0.0f) {
                rowCols[r].push_back(c);
                rowVals[r].push_back(v);
            }
        }
    }

    origRow.resize(rows);
    std::iota(origRow.begin(), origRow.end(), 0);
    std::stable_sort(origRow.begin(), origRow.end(), [&](int i, int j) {
        return rowCols[i].size() > rowCols[j].size();  // longest first
    });

    rowNnzSorted.resize(rows);
    maxRowLen = 0;
    for (int r = 0; r < rows; ++r) {
        rowNnzSorted[r] = static_cast<int>(rowCols[origRow[r]].size());
        maxRowLen = std::max(maxRowLen, rowNnzSorted[r]);
    }

    // count_t = number of sorted rows with length > t (a prefix count,
    // since rowNnzSorted is sorted descending).
    iterPtr.assign(maxRowLen + 1, 0);
    for (int t = 0; t < maxRowLen; ++t) {
        int count_t = 0;
        while (count_t < rows && rowNnzSorted[count_t] > t) ++count_t;
        iterPtr[t + 1] = iterPtr[t] + count_t;
    }

    int totalNnz = iterPtr[maxRowLen];
    colIdx.assign(totalNnz, 0);
    value.assign(totalNnz, 0.0f);
    for (int r = 0; r < rows; ++r) {
        int origR = origRow[r];
        for (int t = 0; t < rowNnzSorted[r]; ++t) {
            int idx = iterPtr[t] + r;
            colIdx[idx] = rowCols[origR][t];
            value[idx] = rowVals[origR][t];
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

float runSpmvJds(const std::vector<int> &colIdx_h, const std::vector<float> &value_h, const std::vector<int> &rowNnzSorted_h,
                  const std::vector<int> &iterPtr_h, const std::vector<int> &origRow_h, int rows,
                  const std::vector<float> &x_h, std::vector<float> &y_h) {
    int totalNnz = static_cast<int>(value_h.size());
    int cols = static_cast<int>(x_h.size());
    int maxRowLen = static_cast<int>(iterPtr_h.size()) - 1;

    int *colIdx_d, *rowNnz_d, *iterPtr_d;
    float *value_d, *x_d, *ySorted_d;
    CUDA_CHECK(cudaMalloc(&colIdx_d, std::max(totalNnz, 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&value_d, std::max(totalNnz, 1) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&rowNnz_d, rows * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&iterPtr_d, (maxRowLen + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&x_d, cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ySorted_d, rows * sizeof(float)));

    if (totalNnz > 0) {
        CUDA_CHECK(cudaMemcpy(colIdx_d, colIdx_h.data(), totalNnz * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(value_d, value_h.data(), totalNnz * sizeof(float), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(rowNnz_d, rowNnzSorted_h.data(), rows * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(iterPtr_d, iterPtr_h.data(), (maxRowLen + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(x_d, x_h.data(), cols * sizeof(float), cudaMemcpyHostToDevice));

    int blockDim = 256;
    int gridDim = (rows + blockDim - 1) / blockDim;

    auto launch = [&]() {
        spmvJdsKernel<<<gridDim, blockDim>>>(colIdx_d, value_d, rowNnz_d, iterPtr_d, rows, x_d, ySorted_d);
        CUDA_CHECK(cudaGetLastError());
    };

    launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    launch();
    float ms = timer.stopAndGetMs();

    std::vector<float> ySorted(rows);
    CUDA_CHECK(cudaMemcpy(ySorted.data(), ySorted_d, rows * sizeof(float), cudaMemcpyDeviceToHost));

    // Undo the row sort: sorted position r corresponds to original row
    // origRow_h[r].
    y_h.assign(rows, 0.0f);
    for (int r = 0; r < rows; ++r) y_h[origRow_h[r]] = ySorted[r];

    CUDA_CHECK(cudaFree(colIdx_d));
    CUDA_CHECK(cudaFree(value_d));
    CUDA_CHECK(cudaFree(rowNnz_d));
    CUDA_CHECK(cudaFree(iterPtr_d));
    CUDA_CHECK(cudaFree(x_d));
    CUDA_CHECK(cudaFree(ySorted_d));
    return ms;
}

bool checkClose(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (!nearlyEqual(a[i], b[i])) return false;
    return true;
}

bool runCase(int rows, int cols, unsigned int seed) {
    std::vector<float> a = generateDenseVariedRowLengths(rows, cols, seed);

    std::vector<int> colIdx, rowNnzSorted, iterPtr, origRow;
    std::vector<float> value;
    int maxRowLen = 0;
    denseToJds(a, rows, cols, colIdx, value, rowNnzSorted, iterPtr, origRow, maxRowLen);

    bool sortedDescending = true;
    for (int r = 1; r < rows; ++r)
        if (rowNnzSorted[r] > rowNnzSorted[r - 1]) sortedDescending = false;

    std::vector<float> x(cols);
    unsigned int state = seed ^ 0xABCDu;
    for (int c = 0; c < cols; ++c) {
        state = state * 1103515245u + 12345u;
        x[c] = static_cast<float>((state >> 8) & 0xFFu) / 25.0f;
    }
    std::vector<float> ref = cpuMatVec(a, rows, cols, x);

    std::vector<float> y;
    float ms = runSpmvJds(colIdx, value, rowNnzSorted, iterPtr, origRow, rows, x, y);

    bool ok = checkClose(y, ref) && sortedDescending;
    printf("%dx%d maxRowLen=%d nnz=%zu (rows %s): %.4f ms  [%s]\n",
           rows, cols, maxRowLen, value.size(), sortedDescending ? "sorted descending" : "UNSORTED", ms, ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    printf("SpMV with JDS format: rows sorted by length, unpadded column-major storage (§17.6):\n");
    bool ok = true;
    ok = runCase(300, 250, 5u) && ok;
    ok = runCase(1024, 512, 123u) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
