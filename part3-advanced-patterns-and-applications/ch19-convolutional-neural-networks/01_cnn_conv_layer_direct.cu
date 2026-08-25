// Chapter 19: Convolutional neural networks
// §19.2, Figs. 19.3-19.7: a direct CUDA convolutional-layer kernel.
//
// Layout (§19.1, Fig. 19.4): input feature maps X are an N*C*H*W tensor (N
// samples in a batch, C input channels, H*W pixels per channel), filters F
// are an M*C*K*K tensor (M output feature maps, one KxK filter per
// input/output channel pair), and output feature maps Y are an
// N*M*H_out*W_out tensor with H_out=H-K+1, W_out=W-K+1 -- a "valid"
// convolution with no padding (§19.1: LeNet-5 simply uses the missing right/
// bottom edge pixels as ghost cells rather than assuming any padding
// convention). No activation function is applied: per §19.1, "the value of
// each output pixel is the sum of convolution results from the
// corresponding patches in all input feature maps."
//
// Thread organization (§19.2, Fig. 19.5/19.6): each thread computes one
// output pixel. 2D TILE_WIDTH x TILE_WIDTH thread blocks each compute one
// tile of one output feature map. The 3D grid is (M, T, N): blockIdx.x
// selects the output feature map (X dimension), blockIdx.z selects the
// sample in the batch (Z dimension), and blockIdx.y linearizes the
// H_grid*W_grid tiles within an output feature map (Y dimension), since only
// one grid dimension is left once X and Z are claimed (Fig. 19.6).

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

#define TILE_WIDTH 16

// Row-major linear index helpers for the N*C*H*W / M*C*K*K / N*M*H_out*W_out
// layouts described above (§19.1, Fig. 19.3 line 1).
__host__ __device__ static inline int idxX(int n, int c, int h, int w, int C, int H, int W) {
    return ((n * C + c) * H + h) * W + w;
}
__host__ __device__ static inline int idxF(int m, int c, int p, int q, int C, int K) {
    return ((m * C + c) * K + p) * K + q;
}
__host__ __device__ static inline int idxY(int n, int m, int h, int w, int M, int H_out, int W_out) {
    return ((n * M + m) * H_out + h) * W_out + w;
}

// ---------------------------------------------------------------------------
// §19.2, Fig. 19.7: ConvLayerForward_Kernel. One thread computes one output
// pixel Y[n,m,h,w]. blockIdx.y is split back into the tile's vertical and
// horizontal position (lines 4-5) using W_grid = W_out/TILE_WIDTH. Each
// thread accumulates a convolution over all C input channels and the KxK
// filter into `acc` (lines 7-12) before writing the final result (line 13).
// This kernel does no shared-memory tiling of X/F, so it is limited by
// global memory bandwidth -- exactly the drawback §19.3 sets out to fix.
// ---------------------------------------------------------------------------
__global__ void convLayerForwardKernel(int C, int W_grid, int K, int H, int W, int M, int H_out, int W_out,
                                        const float *X, const float *F, float *Y) {
    int m = blockIdx.x;
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;
    int n = blockIdx.z;

    float acc = 0.0f;
    for (int c = 0; c < C; ++c) {          // sum over all input channels
        for (int p = 0; p < K; ++p) {      // loop over KxK filter
            for (int q = 0; q < K; ++q) {
                acc += X[idxX(n, c, h + p, w + q, C, H, W)] * F[idxF(m, c, p, q, C, K)];
            }
        }
    }
    Y[idxY(n, m, h, w, M, H_out, W_out)] = acc;
}

// CPU reference, Fig. 19.4: the batched forward path of a convolutional
// layer (same valid-convolution, no-activation semantics as the kernel
// above, computed with plain nested loops over n, m, h, w, then c, p, q).
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

// §19.1, Fig. 19.2(b): the book's own tiny worked example -- C=3 input
// feature maps (3x3 each), M=2 output feature maps, K=2 filters -- used here
// as a hardcoded cross-check of cpuConvLayerForward before trusting it as
// the GPU kernel's reference on larger, randomly generated inputs. Expected
// values read off Fig. 19.2(b): output feature map 0 = [[14,20],[15,24]],
// output feature map 1 = [[12,24],[17,26]] (Y[0,0,0,0]=14 is also spelled
// out in Eq. (19.1)).
bool checkBookExample() {
    const int N = 1, C = 3, H = 3, W = 3, M = 2, K = 2;
    std::vector<float> X = {
        // channel 0
        1, 2, 0,
        1, 1, 3,
        0, 2, 2,
        // channel 1
        0, 2, 1,
        0, 3, 2,
        1, 1, 0,
        // channel 2
        1, 2, 1,
        0, 1, 3,
        3, 3, 2};
    std::vector<float> F = {
        // output map 0: F0,0 F0,1 F0,2
        1, 1, 2, 2,
        1, 1, 1, 1,
        0, 1, 1, 0,
        // output map 1: F1,0 F1,1 F1,2
        1, 0, 0, 1,
        2, 1, 2, 1,
        1, 2, 2, 0};
    std::vector<float> expected = {14, 20, 15, 24,
                                    12, 24, 17, 26};
    std::vector<float> y = cpuConvLayerForward(X, N, C, H, W, F, M, K);
    bool ok = checkClose(y, expected);
    printf("book example (§19.1, Fig. 19.2b): C=3 3x3, K=2, M=2 -> Y[0,0,0,0]=%.0f [%s]\n", y[0], ok ? "match" : "MISMATCH");
    return ok;
}

float runDirectConv(const std::vector<float> &X_h, int N, int C, int H, int W,
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

    // §19.2, Fig. 19.5: host launch code -- W_grid/H_grid tile counts,
    // T = H_grid*W_grid linearized tiles per feature map, gridDim(M, T, N).
    int W_grid = W_out / TILE_WIDTH;
    int H_grid = H_out / TILE_WIDTH;
    int T = H_grid * W_grid;
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(M, T, N);

    auto launch = [&]() {
        convLayerForwardKernel<<<gridDim, blockDim>>>(C, W_grid, K, H, W, M, H_out, W_out, X_d, F_d, Y_d);
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
    float ms = runDirectConv(X, N, C, H, W, F, M, K, y);

    bool ok = checkClose(y, ref);
    int H_out = H - K + 1, W_out = W - K + 1;
    printf("%s: N=%d C=%d H=%d W=%d K=%d M=%d -> H_out=%d W_out=%d: %.4f ms  [%s]\n",
           label, N, C, H, W, K, M, H_out, W_out, ms, ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    printf("Direct CUDA convolutional layer kernel, one thread per output pixel (§19.2, Fig. 19.7):\n");
    bool ok = true;
    ok = checkBookExample() && ok;
    // Mirrors Fig. 19.6's own grid example exactly: M=4 output feature
    // maps, each output feature map tiled as a 2x2 grid of TILE_WIDTH=16
    // tiles (H_out=W_out=32).
    ok = runCase("Fig 19.6 grid example (2x2 tiles/map)", 2, 3, 36, 36, 4, 5, 42u) && ok;
    // A single TILE_WIDTH tile per output feature map (T=1).
    ok = runCase("single-tile config", 1, 2, 18, 18, 6, 3, 7u) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
