// Chapter 11: Scan
// §11.10  Parallel scan with the Brent-Kung algorithm (Fig. 11.19, 11.20)
//
// IMPORTANT DISCLOSURE: unlike every other section in this chapter, §11.10
// gives NO CODE FIGURE for the Brent-Kung scan kernel. The book presents
// the algorithm only as two diagrams (Fig. 11.19: the up-sweep reduction
// tree and down-sweep reverse tree; Fig. 11.20: the per-position state
// table across levels of the reverse tree) plus a prose walk-through and an
// operation-count derivation, then says explicitly: "We leave the
// implementation of the Brent-Kung algorithm for parallel scan as an
// exercise for the reader." The kernel below is THIS FILE's own
// implementation of the algorithm exactly as the book describes and
// diagrams it (verified step-by-step against the book's own N=16 worked
// example below), not a transcription of book source code -- there is none
// to transcribe.
//
// The algorithm, per §11.10:
//   Phase 1 (reduction / up-sweep, top half of Fig. 11.19): build a sum in
//   log2(N) steps. Step k updates only positions of the form
//   (t+1)*2*stride - 1 for stride = 2^k, i.e. positions 2n-1, then 4n-1,
//   then 8n-1, etc. For N=16 this takes 4 steps and 8+4+2+1 = 15 = N-1
//   operations total (matching the general formula N/2+N/4+...+1 = N-1).
//
//   Phase 2 (reverse / down-sweep tree, bottom half of Fig. 11.19 /
//   Fig. 11.20): distribute partial sums back out so every position
//   accumulates the full range of inputs at or before it. Levels run with
//   stride N/4, N/8, ..., 1 (log2(N)-1 levels for N=16: 3 levels), each
//   level updating position (t+1)*2*stride-1+stride when in range. The
//   book's own N=16 example is reproduced exactly by this addressing:
//     stride=4: buffer[11] += buffer[7]                      (1 op)
//     stride=2: buffer[5,9,13] += buffer[3,7,11]              (3 ops)
//     stride=1: buffer[2,4,6,8,10,12,14] += buffer[1,3,5,7,9,11,13]  (7 ops)
//   totaling 1+3+7 = 11 = N-1-log2(N) operations, matching the book's
//   stated "16/8-1 + 16/4-1 + 16/2-1 = 11" exactly.
//
// Total operations: (N-1) + (N-1-log2(N)) = 2N-2-log2(N), which is O(N) --
// better work efficiency than Kogge-Stone's O(N*log2(N)) (N*log2(N)-(N-1)
// operations, §11.5). But per §11.10's own conclusion, this file-06-style
// work-efficiency argument mostly matters BEFORE thread coarsening is
// applied (coarsening already pushes most work onto efficient sequential
// per-thread scans, per §11.6); the book states that at the warp level,
// Kogge-Stone in practice outperforms Brent-Kung, because Brent-Kung's
// "saved" work is replaced by inactive-but-still-resource-consuming SIMD
// lanes. This file measures the two kernels' operation counts (from the
// book's own formulas, not invented numbers) and their wall-clock time
// against each other for a genuine, fairly-warmed-up, same-launch-config
// comparison -- see main() -- without asserting a specific winner beyond
// what is actually measured.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>

#include "../../common/cuda_utils.h"

// ---------------------------------------------------------------------------
// §11.10, Fig. 11.19/11.20 (no code figure in the book -- see file header):
// Brent-Kung inclusive scan, per block segment (blockDim.x elements/block,
// matching this chapter's other per-block kernels).
// ---------------------------------------------------------------------------
__global__ void brentKungScanKernel(const float *input, float *output, unsigned int N) {
    extern __shared__ float buffer_s[];

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    buffer_s[threadIdx.x] = (i < N) ? input[i] : 0.0f;

    // Phase 1: reduction (up-sweep) tree.
    for (unsigned int stride = 1; stride <= blockDim.x / 2; stride *= 2) {
        __syncthreads();
        unsigned int index = (threadIdx.x + 1) * 2 * stride - 1;
        if (index < blockDim.x) {
            buffer_s[index] += buffer_s[index - stride];
        }
    }

    // Phase 2: reverse (down-sweep) distribution tree.
    for (unsigned int stride = blockDim.x / 4; stride >= 1; stride /= 2) {
        __syncthreads();
        unsigned int index = (threadIdx.x + 1) * 2 * stride - 1;
        if (index + stride < blockDim.x) {
            buffer_s[index + stride] += buffer_s[index];
        }
    }
    __syncthreads();

    if (i < N) {
        output[i] = buffer_s[threadIdx.x];
    }
}

// File 01's single-buffer Kogge-Stone kernel, duplicated here (this
// project's self-contained-file convention) so this file can run a direct,
// same-launch-configuration timing comparison against Brent-Kung.
__global__ void koggeStoneScanKernel(const float *input, float *output, unsigned int N) {
    extern __shared__ float buffer_s[];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    buffer_s[threadIdx.x] = (i < N) ? input[i] : 0.0f;

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        float temp;
        if (threadIdx.x >= stride) {
            temp = buffer_s[threadIdx.x] + buffer_s[threadIdx.x - stride];
        }
        __syncthreads();
        if (threadIdx.x >= stride) {
            buffer_s[threadIdx.x] = temp;
        }
    }

    if (i < N) {
        output[i] = buffer_s[threadIdx.x];
    }
}

void scanSegmentsCPU(const float *input, float *output, unsigned int N, unsigned int segSize) {
    for (unsigned int base = 0; base < N; base += segSize) {
        unsigned int end = std::min(base + segSize, N);
        float acc = 0.0f;
        for (unsigned int i = base; i < end; ++i) {
            acc += input[i];
            output[i] = acc;
        }
    }
}

std::vector<float> generateInput(unsigned int n) {
    std::vector<float> v(n);
    unsigned int state = 723456789u;
    for (unsigned int i = 0; i < n; ++i) {
        state = state * 1103515245u + 12345u;
        v[i] = static_cast<float>((state >> 8) & 0xFFFFFF) / static_cast<float>(0x1000000);
    }
    return v;
}

// §11.5/§11.10's own operation-count formulas (not measured -- derived
// directly from the book's text), for a segment of size segSize.
double koggeStoneOps(unsigned int segSize) {
    double N = static_cast<double>(segSize);
    return N * std::log2(N) - (N - 1.0);
}
double brentKungOps(unsigned int segSize) {
    double N = static_cast<double>(segSize);
    return 2.0 * N - 2.0 - std::log2(N);
}

// Runs both kernels (each individually warmed up before its own timed
// launch, per this project's timing-fairness convention for in-process
// kernel comparisons) on identical input with identical launch geometry.
void runComparison(unsigned int segSize, unsigned int numBlocks, bool *ok) {
    unsigned int n = segSize * numBlocks;
    std::vector<float> input_h = generateInput(n);

    std::vector<float> ref(n);
    scanSegmentsCPU(input_h.data(), ref.data(), n, segSize);

    size_t bytes = n * sizeof(float);
    size_t shmemBytes = segSize * sizeof(float);
    dim3 dimBlock(segSize);
    dim3 dimGrid(numBlocks);

    float *input_d, *bkOutput_d, *ksOutput_d;
    CUDA_CHECK(cudaMalloc((void **)&input_d, bytes));
    CUDA_CHECK(cudaMalloc((void **)&bkOutput_d, bytes));
    CUDA_CHECK(cudaMalloc((void **)&ksOutput_d, bytes));
    CUDA_CHECK(cudaMemcpy(input_d, input_h.data(), bytes, cudaMemcpyHostToDevice));

    // Brent-Kung: warm-up, then timed.
    brentKungScanKernel<<<dimGrid, dimBlock, shmemBytes>>>(input_d, bkOutput_d, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    GpuTimer bkTimer;
    bkTimer.start();
    brentKungScanKernel<<<dimGrid, dimBlock, shmemBytes>>>(input_d, bkOutput_d, n);
    CUDA_CHECK(cudaGetLastError());
    float bkMs = bkTimer.stopAndGetMs();

    // Kogge-Stone: warm-up, then timed (same input, same launch geometry).
    koggeStoneScanKernel<<<dimGrid, dimBlock, shmemBytes>>>(input_d, ksOutput_d, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    GpuTimer ksTimer;
    ksTimer.start();
    koggeStoneScanKernel<<<dimGrid, dimBlock, shmemBytes>>>(input_d, ksOutput_d, n);
    CUDA_CHECK(cudaGetLastError());
    float ksMs = ksTimer.stopAndGetMs();

    std::vector<float> bkOutput_h(n), ksOutput_h(n);
    CUDA_CHECK(cudaMemcpy(bkOutput_h.data(), bkOutput_d, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ksOutput_h.data(), ksOutput_d, bytes, cudaMemcpyDeviceToHost));

    bool bkOk = true, ksOk = true;
    for (unsigned int i = 0; i < n; ++i) {
        if (!nearlyEqual(bkOutput_h[i], ref[i], 1e-2f)) {
            bkOk = false;
            printf("  Brent-Kung mismatch at %u: cpu=%.6f gpu=%.6f\n", i, ref[i], bkOutput_h[i]);
        }
        if (!nearlyEqual(ksOutput_h[i], ref[i], 1e-2f)) {
            ksOk = false;
            printf("  Kogge-Stone mismatch at %u: cpu=%.6f gpu=%.6f\n", i, ref[i], ksOutput_h[i]);
        }
    }

    double bkOps = brentKungOps(segSize);
    double ksOps = koggeStoneOps(segSize);
    printf("segSize=%u blocks=%u N=%u:\n", segSize, numBlocks, n);
    printf("  Brent-Kung : last=%.6f  %.4f ms  ops/segment(book formula 2N-2-log2N)=%.1f  [%s]\n",
           bkOutput_h[n - 1], bkMs, bkOps, bkOk ? "match" : "MISMATCH");
    printf("  Kogge-Stone: last=%.6f  %.4f ms  ops/segment(book formula N*log2N-(N-1))=%.1f  [%s]\n",
           ksOutput_h[n - 1], ksMs, ksOps, ksOk ? "match" : "MISMATCH");

    CUDA_CHECK(cudaFree(input_d));
    CUDA_CHECK(cudaFree(bkOutput_d));
    CUDA_CHECK(cudaFree(ksOutput_d));

    *ok = *ok && bkOk && ksOk;
}

int main() {
    bool ok = true;
    runComparison(64, 4, &ok);
    runComparison(256, 8, &ok);
    runComparison(1024, 4, &ok);

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
