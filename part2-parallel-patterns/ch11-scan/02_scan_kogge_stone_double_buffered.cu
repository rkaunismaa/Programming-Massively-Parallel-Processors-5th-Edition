// Chapter 11: Scan
// §11.3  Double-buffering to reduce synchronization (Fig. 11.4, Fig. 11.5)
//
// File 01's kernel pays two __syncthreads() calls per loop iteration: one
// for a true dependence (wait for writes) and one for a false dependence
// (wait for reads before the same-iteration overwrite). §11.3's insight:
// false dependences can be eliminated entirely by never writing and reading
// the SAME buffer in the same iteration. We keep two shared-memory buffers,
// alternating which one is "input" and which is "output" each iteration --
// once an iteration's outputs are computed, that iteration's inputs are
// dead and the buffer they lived in becomes the NEXT iteration's output
// buffer. Because active threads read from inBuffer_s and write to
// outBuffer_s, thread i+stride's read of inBuffer_s[i] can never be
// clobbered by thread i's write to outBuffer_s[i] -- they're different
// memory locations even though both are logically "position i". Only one
// __syncthreads() per iteration remains, for the true dependence.
//
// One wrinkle from Fig. 11.4/11.5: values that are already final (e.g. x0,
// which never changes after iteration 0) are NOT automatically retained
// across the buffer swap the way they were in the single-buffer version --
// they must be explicitly copied from inBuffer_s to outBuffer_s by the
// threads that would otherwise sit idle (threadIdx.x < stride).

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#include "../../common/cuda_utils.h"

// ---------------------------------------------------------------------------
// §11.3, Fig. 11.5: double-buffered Kogge-Stone inclusive scan, per block
// segment. Shared memory holds two segSize-element buffers back to back.
// ---------------------------------------------------------------------------
__global__ void koggeStoneDoubleBufferedScanKernel(const float *input, float *output, unsigned int N) {
    extern __shared__ float sharedMem[];
    float *buffer1_s = sharedMem;
    float *buffer2_s = sharedMem + blockDim.x;
    float *inBuffer_s = buffer1_s;
    float *outBuffer_s = buffer2_s;

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    inBuffer_s[threadIdx.x] = (i < N) ? input[i] : 0.0f;

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();  // true dependence only -- no false dependence to guard
        if (threadIdx.x >= stride) {
            outBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x] + inBuffer_s[threadIdx.x - stride];
        } else {
            // Already-final values must be explicitly carried into the new
            // output buffer -- they are not retained automatically.
            outBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x];
        }
        float *tmp = inBuffer_s;
        inBuffer_s = outBuffer_s;
        outBuffer_s = tmp;
    }

    if (i < N) {
        output[i] = inBuffer_s[threadIdx.x];
    }
}

// Sequential reference, per segment (see file 01 for rationale).
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
    unsigned int state = 223456789u;
    for (unsigned int i = 0; i < n; ++i) {
        state = state * 1103515245u + 12345u;
        v[i] = static_cast<float>((state >> 8) & 0xFFFFFF) / static_cast<float>(0x1000000);
    }
    return v;
}

float runDoubleBufferedScan(const std::vector<float> &input_h, std::vector<float> &output_h,
                             unsigned int segSize) {
    unsigned int n = static_cast<unsigned int>(input_h.size());
    size_t bytes = n * sizeof(float);
    size_t shmemBytes = 2 * segSize * sizeof(float);

    float *input_d, *output_d;
    CUDA_CHECK(cudaMalloc((void **)&input_d, bytes));
    CUDA_CHECK(cudaMalloc((void **)&output_d, bytes));
    CUDA_CHECK(cudaMemcpy(input_d, input_h.data(), bytes, cudaMemcpyHostToDevice));

    dim3 dimBlock(segSize);
    dim3 dimGrid(n / segSize);

    koggeStoneDoubleBufferedScanKernel<<<dimGrid, dimBlock, shmemBytes>>>(input_d, output_d, n);  // warm-up
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    koggeStoneDoubleBufferedScanKernel<<<dimGrid, dimBlock, shmemBytes>>>(input_d, output_d, n);
    CUDA_CHECK(cudaGetLastError());
    float ms = timer.stopAndGetMs();

    output_h.resize(n);
    CUDA_CHECK(cudaMemcpy(output_h.data(), output_d, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(input_d));
    CUDA_CHECK(cudaFree(output_d));
    return ms;
}

bool runTestCase(unsigned int segSize, unsigned int numBlocks) {
    unsigned int n = segSize * numBlocks;
    std::vector<float> input_h = generateInput(n);

    std::vector<float> ref(n);
    scanSegmentsCPU(input_h.data(), ref.data(), n, segSize);

    std::vector<float> gpu;
    float ms = runDoubleBufferedScan(input_h, gpu, segSize);

    bool ok = true;
    for (unsigned int i = 0; i < n; ++i) {
        if (!nearlyEqual(gpu[i], ref[i], 1e-2f)) {
            ok = false;
            printf("  mismatch at %u: cpu=%.6f gpu=%.6f\n", i, ref[i], gpu[i]);
            break;
        }
    }
    printf("segSize=%u blocks=%u N=%u: last=%.6f (cpu) / %.6f (gpu)  %.4f ms  [%s]\n",
           segSize, numBlocks, n, ref[n - 1], gpu[n - 1], ms, ok ? "match" : "MISMATCH");
    return ok;
}

int main() {
    bool ok = true;
    ok = runTestCase(64, 4) && ok;
    ok = runTestCase(256, 8) && ok;
    ok = runTestCase(1024, 4) && ok;

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
