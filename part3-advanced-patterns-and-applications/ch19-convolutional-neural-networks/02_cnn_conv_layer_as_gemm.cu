// Chapter 19: Convolutional neural networks
// §19.3, Figs. 19.8-19.11: formulating a convolutional layer as GEMM.
//
// §19.3's central idea (Chellapilla et al. [8]): each output feature map
// element is a dot product between a linearized filter bank row and a
// linearized patch of input feature map pixels, so the whole convolutional
// layer is one matrix multiplication F * B, where:
//   - F is an M x (C*K*K) matrix -- the filter bank array is already laid
//     out this way in memory (§19.3: "there is no rearrangement needed
//     inside these filter banks"), since F's M*C*K*K layout already puts
//     output-feature-map index as the highest dimension.
//   - B is a (C*K*K) x (H_out*W_out) matrix formed by unfolding and
//     duplicating input feature map patches -- one column per output pixel.
//
// §19.3 then shows that explicitly materializing B is wasteful (Eq. 19.4:
// an expansion ratio of K^2*H_out*W_out/(H_in*W_in) over X, easily 20x or
// more for real layer sizes) and not worth the extra global memory traffic,
// motivating an *implicit* approach: a tiled matrix-multiplication kernel
// (adapted from Fig. 5.9) that loads each B tile on demand straight from X
// via the mapping in Eq. (19.5), never materializing B in global memory.
// Fig. 19.11's ConvLayer_MM_Kernel is exactly this kernel, implemented
// below unchanged (including its assumption -- stated in the book -- that
// dimensions divide evenly by TILE_WIDTH, so no bounds checking is done).

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

#define TILE_WIDTH 16

// Row-major linear index helpers, same N*C*H*W / M*C*K*K / N*M*H_out*W_out
// layouts as §19.1 (Fig. 19.3 line 1) -- kept as this file's own copy per
// this repo's no-cross-file-includes convention.
static inline int idxX(int n, int c, int h, int w, int C, int H, int W) {
    return ((n * C + c) * H + h) * W + w;
}
static inline int idxF(int m, int c, int p, int q, int C, int K) {
    return ((m * C + c) * K + p) * K + q;
}
static inline int idxY(int n, int m, int h, int w, int M, int H_out, int W_out) {
    return ((n * M + m) * H_out + h) * W_out + w;
}

// ---------------------------------------------------------------------------
// §19.3, Fig. 19.11: ConvLayer_MM_Kernel -- a tiled matrix multiplication
// (adapted from the Chapter 5 tiled MM kernel) that computes Y = F * B one
// sample (bz) at a time, without ever materializing B. Row indexes into F's
// M dimension (which output feature map); Col indexes into B's H_out*W_out
// columns (which output pixel). Each phase `ph` loads one TILE_WIDTH-wide
// slice of F directly (line 23) and one TILE_WIDTH-wide slice of the
// *conceptual* B matrix by mapping each B[u,v] element back to its source
// X element via Eq. (19.5) (line 28), before the usual tiled-MM inner
// product accumulation into Pvalue.
// ---------------------------------------------------------------------------
__global__ void convLayerMMKernel(int C, int M, int H, int W, int K, const float *F, const float *X, float *Y) {
    __shared__ float Fds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int bz = blockIdx.z;  // bz is used for sample index
    int tx = threadIdx.x, ty = threadIdx.y;

    // Identify the row/column of the Y element to work on.
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    int W_out = W - K + 1;

    // Loop over the F and B tiles for the Y element.
    float Pvalue = 0.0f;
    for (int ph = 0; ph < (C * K * K) / TILE_WIDTH; ++ph) {
        // C*K*K is the width of the F matrix.

        // Load F and B tiles into shared memory.
        Fds[ty][tx] = F[Row * (C * K * K) + (ph * TILE_WIDTH + tx)];

        int u = ph * TILE_WIDTH + ty;
        int v = Col;

        // Eq. (19.5): B[u,v] <- X[n, u/(K*K), (u%(K*K))/K + v/W_out, (u%(K*K))%K + v%W_out]
        Bds[ty][tx] = X[bz * (C * H * W) + (u / (K * K)) * (H * W) +
                        ((u % (K * K)) / K + v / W_out) * W +
                        ((u % (K * K)) % K + v % W_out)];

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += Fds[ty][k] * Bds[k][tx];

        __syncthreads();
    }
    Y[bz * M * (H - K + 1) * (W - K + 1) + Row * (H - K + 1) * (W - K + 1) + Col] = Pvalue;
}

// CPU reference: same batched forward convolutional layer as file 01, kept
// as this file's own independent copy (no cross-file includes).
std::vector<float> cpuConvLayerForward(const std::vector<float> &X, int N, int C, int H, int W,
                                        const std::vector<float> &F, int M, int K) {
    int H_out = H - K + 1, W_out = W - K + 1;
    std::vector<float> Y(static_cast<size_t>(N) * M * H_out * W_out, 0.0f);
    for (int n = 0; n < N; ++n)
        for (int m = 0; m < M; ++m)
            for (int h = 0; h < H_out; ++h)
                for (int w = 0; w < W_out; ++w) {
                    float acc = 0.0f;
                    for (int c = 0; c < C; ++c)
                        for (int p = 0; p < K; ++p)
                            for (int q = 0; q < K; ++q)
                                acc += X[idxX(n, c, h + p, w + q, C, H, W)] * F[idxF(m, c, p, q, C, K)];
                    Y[idxY(n, m, h, w, M, H_out, W_out)] = acc;
                }
    return Y;
}

// §19.3, "Cost of Explicitly Unfolding Input Feature Maps" + Eq. (19.5):
// explicitly materialize one sample's B matrix (C*K*K rows x H_out*W_out
// columns) by unfolding and duplicating patches of X -- the expensive
// approach the chapter costs out before motivating the implicit, kernel-
// internal unfolding done by convLayerMMKernel above.
std::vector<float> unfoldToB(const std::vector<float> &X, int n, int C, int H, int W, int K) {
    int H_out = H - K + 1, W_out = W - K + 1;
    int rows = C * K * K, cols = H_out * W_out;
    std::vector<float> B(static_cast<size_t>(rows) * cols);
    for (int u = 0; u < rows; ++u) {
        int c = u / (K * K);
        int p = (u % (K * K)) / K;
        int q = (u % (K * K)) % K;
        for (int v = 0; v < cols; ++v) {
            int h0 = v / W_out, w0 = v % W_out;
            B[static_cast<size_t>(u) * cols + v] = X[idxX(n, c, p + h0, q + w0, C, H, W)];
        }
    }
    return B;
}

// Plain CPU matrix multiply: F (M x inner) * B (inner x cols) -> Y (M x cols),
// standing in for the GEMM that §19.3 says a convolutional layer reduces to.
std::vector<float> cpuMatMul(const std::vector<float> &F, const std::vector<float> &B, int M, int inner, int cols) {
    std::vector<float> Y(static_cast<size_t>(M) * cols, 0.0f);
    for (int r = 0; r < M; ++r)
        for (int cc = 0; cc < cols; ++cc) {
            float acc = 0.0f;
            for (int k = 0; k < inner; ++k)
                acc += F[static_cast<size_t>(r) * inner + k] * B[static_cast<size_t>(k) * cols + cc];
            Y[static_cast<size_t>(r) * cols + cc] = acc;
        }
    return Y;
}

std::vector<float> randomVec(size_t n, unsigned int seed) {
    std::vector<float> v(n);
    unsigned int state = seed;
    auto nextRand = [&]() -> float {
        state = state * 1103515245u + 12345u;
        return static_cast<float>((state >> 8) & 0xFFFFFFu) / static_cast<float>(0xFFFFFFu);  // [0,1)
    };
    for (size_t i = 0; i < n; ++i) v[i] = 2.0f * nextRand() - 1.0f;  // [-1,1)
    return v;
}

bool checkClose(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (!nearlyEqual(a[i], b[i])) return false;
    return true;
}

// §19.3, Fig. 19.8: the book's own tiny worked example -- same C=3, 3x3
// input feature maps, M=2 output feature maps, K=2 filters as file 01's
// §19.1 example. This checks two things against the book's stated values
// (Y[0,0,0,0]=14 from Eq. 19.2/19.3, full maps from Fig. 19.2b): (1) that
// this file's own CPU reference agrees, and (2) that explicitly unfolding X
// into B (Eq. 19.5) and multiplying F*B on the CPU gives the identical
// result -- i.e. the GEMM reformulation itself is correct, independent of
// the tiled GPU kernel. Also reports the B-matrix expansion ratio (Eq.
// 19.4): the book's own example works out to 48/27 = 16/9 ~= 1.78x.
bool checkBookExample() {
    const int N = 1, C = 3, H = 3, W = 3, M = 2, K = 2;
    std::vector<float> X = {
        1, 2, 0,
        1, 1, 3,
        0, 2, 2,

        0, 2, 1,
        0, 3, 2,
        1, 1, 0,

        1, 2, 1,
        0, 1, 3,
        3, 3, 2};
    std::vector<float> F = {
        1, 1, 2, 2,
        1, 1, 1, 1,
        0, 1, 1, 0,

        1, 0, 0, 1,
        2, 1, 2, 1,
        1, 2, 2, 0};
    std::vector<float> expected = {14, 20, 15, 24,
                                    12, 24, 17, 26};

    std::vector<float> yDirect = cpuConvLayerForward(X, N, C, H, W, F, M, K);
    bool directOk = checkClose(yDirect, expected);

    int H_out = H - K + 1, W_out = W - K + 1;
    std::vector<float> B = unfoldToB(X, 0, C, H, W, K);
    std::vector<float> yGemm = cpuMatMul(F, B, M, C * K * K, H_out * W_out);
    bool gemmOk = checkClose(yGemm, expected);

    float ratio = static_cast<float>(C * K * K * H_out * W_out) / static_cast<float>(C * H * W);
    bool ok = directOk && gemmOk;
    printf("book example (§19.3, Fig. 19.8): B is %dx%d (expansion ratio %.3fx over X, Eq. 19.4) -> Y[0,0,0,0]=%.0f [%s]\n",
           C * K * K, H_out * W_out, ratio, yGemm[0], ok ? "match" : "MISMATCH");
    return ok;
}

float runGemmConv(const std::vector<float> &X_h, int N, int C, int H, int W,
                   const std::vector<float> &F_h, int M, int K, std::vector<float> &Y_h) {
    int H_out = H - K + 1, W_out = W - K + 1;
    size_t xSize = X_h.size(), fSize = F_h.size();
    size_t ySize = static_cast<size_t>(N) * M * H_out * W_out;

    float *X_d, *F_d, *Y_d;
    CUDA_CHECK(cudaMalloc(&X_d, xSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&F_d, fSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Y_d, ySize * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(X_d, X_h.data(), xSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(F_d, F_h.data(), fSize * sizeof(float), cudaMemcpyHostToDevice));

    // Row = by*TILE_WIDTH+ty ranges over F's M rows; Col = bx*TILE_WIDTH+tx
    // ranges over B's H_out*W_out columns; bz ranges over the N samples.
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim((H_out * W_out) / TILE_WIDTH, M / TILE_WIDTH, N);

    auto launch = [&]() {
        convLayerMMKernel<<<gridDim, blockDim>>>(C, M, H, W, K, F_d, X_d, Y_d);
        CUDA_CHECK(cudaGetLastError());
    };

    launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    launch();
    float ms = timer.stopAndGetMs();

    Y_h.resize(ySize);
    CUDA_CHECK(cudaMemcpy(Y_h.data(), Y_d, ySize * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(X_d));
    CUDA_CHECK(cudaFree(F_d));
    CUDA_CHECK(cudaFree(Y_d));
    return ms;
}

bool runCase(const char *label, int N, int C, int H, int W, int M, int K, unsigned int seed) {
    std::vector<float> X = randomVec(static_cast<size_t>(N) * C * H * W, seed);
    std::vector<float> F = randomVec(static_cast<size_t>(M) * C * K * K, seed ^ 0x9E3779B9u);

    std::vector<float> ref = cpuConvLayerForward(X, N, C, H, W, F, M, K);

    std::vector<float> y;
    float ms = runGemmConv(X, N, C, H, W, F, M, K, y);

    bool ok = checkClose(y, ref);
    int H_out = H - K + 1, W_out = W - K + 1;
    int phases = (C * K * K) / TILE_WIDTH;
    printf("%s: N=%d C=%d H=%d W=%d K=%d M=%d -> H_out=%d W_out=%d, %d phase(s): %.4f ms  [%s]\n",
           label, N, C, H, W, K, M, H_out, W_out, phases, ms, ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    printf("Convolutional layer as GEMM: tiled matrix multiplication with implicit B-matrix unfolding (§19.3, Fig. 19.11):\n");
    bool ok = true;
    ok = checkBookExample() && ok;
    // Multi-phase config: C*K*K=48 -> 3 tiled-MM phases; H_out*W_out=64 (4
    // Col tiles), M=32 (2 Row tiles).
    ok = runCase("multi-phase config", 2, 3, 11, 11, 32, 4, 42u) && ok;
    // Single-phase, single-tile config: C*K*K=16 -> 1 phase; H_out*W_out=16
    // and M=16, one tile each.
    ok = runCase("single-tile config", 1, 1, 7, 7, 16, 4, 7u) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
