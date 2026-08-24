// Chapter 4: Compute architecture and scheduling
// §4.5  Control divergence
//
// §4.5 explains that an SM executes all 32 threads of a warp in lockstep
// (SIMD/SIMT). When threads within a warp take different control-flow
// paths -- e.g. some take the then-path of an if-else and others take the
// else-path -- the hardware makes multiple passes through the warp: one
// pass with the then-path threads active and the else-path threads masked
// off, then another pass with the roles reversed (Fig. 4.9). "[T]he cost
// of divergence... is the extra passes the hardware needs to take... as
// well as the execution resources that are consumed by the inactive
// threads in each pass" (§4.5). §4.5's own text illustrates a
// thread-index-dependent divergent condition with `if(threadIdx.x > 2)`;
// this sample instead uses `threadIdx.x % 2 == 0`, the condition §4.3's
// Fig. 4.4 uses for its (incorrect-__syncthreads()) divergence example --
// like `> 2`, its decision is based on threadIdx.x, so by §4.5's own test
// ("If the decision condition is based on threadIdx values, the control
// statement can potentially cause thread divergence") it is exactly the
// kind of condition §4.5 is about, and since threadIdx.x parity alternates
// every thread, every warp that takes this branch is guaranteed to
// diverge (a stronger, easier-to-reason-about guarantee than `> 2`, which
// only diverges the warps straddling the threshold).
//
// This sample runs two kernels over the same input array:
//   - divergentKernel: branches on `threadIdx.x % 2` (checkerboard --
//     touches every warp) and runs one of two *different* per-element
//     formulas depending on the branch taken.
//   - divergenceFreeKernel: computes the identical per-element mapping
//     (even global index -> formula A, odd global index -> formula B) but
//     with no thread-index-dependent branch at all. It arithmetically
//     selects (once, before the loop) *which single factor* to multiply
//     by, so every thread in every warp executes the same instruction
//     stream and does exactly the same number of multiplies as
//     divergentKernel -- the only structural difference between the two
//     kernels is whether the SIMD hardware has to take one pass or two.
//
// Because blockDim.x (256) is even, blockIdx.x*blockDim.x is always even,
// so threadIdx.x % 2 == (blockIdx.x*blockDim.x + threadIdx.x) % 2, i.e.
// threadIdx.x parity and global-index parity coincide here -- both
// kernels therefore apply formula A to exactly the even-indexed elements
// and formula B to exactly the odd-indexed elements, matching a single
// CPU reference. Both formulas are pure repeated multiplications (no
// fused multiply-add), so host and device arithmetic follow the same
// IEEE-754 instruction sequence with no FMA-contraction ambiguity, and
// results are expected to match closely (checked with nearlyEqual).
//
// Correctness (PASS/FAIL) is judged solely by whether both kernels agree
// with the CPU reference -- not by which kernel is faster. Both kernels
// are launched once beforehand (results discarded) as a warm-up, so the
// one-time PTX->SASS JIT compilation cost the driver pays the first time
// a kernel is touched on this GPU (whose SASS this build doesn't ship
// natively for, since we compile for -arch=sm_75) doesn't leak into
// whichever kernel happens to be timed first. Both kernels' warm timings
// are then printed so divergence's steady-state cost is visible.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../common/cuda_utils.h"

// Number of sequential multiply steps per element. Large enough that any
// per-pass / per-branch overhead accumulates to a measurable amount of
// GPU time, not so large that the demo takes long to run.
constexpr int ITERS = 4000;

// Pure-multiplication "formulas". No addition is combined with the
// multiply in the same expression, so there is no fused multiply-add
// opportunity for the compiler to exploit differently on host vs device.
constexpr float FACTOR_A = 1.0000001f;  // even-index formula
constexpr float FACTOR_B = 0.9999999f;  // odd-index formula

// ---------------------------------------------------------------------------
// §4.5, Fig. 4.9: divergent if-else. threadIdx.x % 2 alternates within
// every warp, so every warp executing this kernel takes two passes: one
// with the even lanes active (applying FACTOR_A) and the odd lanes
// masked off, and one with the odd lanes active (applying FACTOR_B) and
// the even lanes masked off.
// ---------------------------------------------------------------------------
__global__ void divergentKernel(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        if (threadIdx.x % 2 == 0) {
            for (int k = 0; k < ITERS; ++k) {
                x *= FACTOR_A;
            }
        } else {
            for (int k = 0; k < ITERS; ++k) {
                x *= FACTOR_B;
            }
        }
        out[i] = x;
    }
}

// ---------------------------------------------------------------------------
// Divergence-free equivalent: no thread-index-dependent branch. The
// per-thread factor is chosen once, arithmetically (not via a branch),
// before the loop; the loop body itself is then a single multiply, same
// as each side of divergentKernel's if-else, so both kernels do the same
// total number of multiplies per thread. Every thread in every warp runs
// the identical loop with the identical instructions -- only the
// (arithmetically selected) factor value differs between lanes, which is
// the essence of true SIMD/SIMT execution with a single pass.
// ---------------------------------------------------------------------------
__global__ void divergenceFreeKernel(const float *in, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        // 1.0f for even threadIdx.x, 0.0f for odd -- computed arithmetically,
        // not via a branch.
        float isEven = static_cast<float>((threadIdx.x & 1) == 0);
        // isEven is exactly 1.0f or 0.0f, so this select is mathematically
        // exactly FACTOR_A or exactly FACTOR_B (modulo any FMA contraction
        // the compiler applies to this expression, which nearlyEqual's
        // tolerance absorbs) -- matching whichever factor the divergent
        // kernel's branch would have used for this thread. Computed once,
        // outside the loop, so the loop itself contains a
        // single multiply per iteration, matching divergentKernel's
        // per-branch body exactly in per-thread arithmetic cost.
        float factor = isEven * FACTOR_A + (1.0f - isEven) * FACTOR_B;
        for (int k = 0; k < ITERS; ++k) {
            x *= factor;
        }
        out[i] = x;
    }
}

// CPU reference: same two formulas, applied by global-index parity (which
// coincides with threadIdx.x parity for our even block size).
void referenceHost(const float *in, float *out, int n) {
    for (int i = 0; i < n; ++i) {
        float x = in[i];
        float factor = (i % 2 == 0) ? FACTOR_A : FACTOR_B;
        for (int k = 0; k < ITERS; ++k) {
            x *= factor;
        }
        out[i] = x;
    }
}

int main() {
    const int n = 1 << 20;  // ~1M elements
    const int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;

    std::vector<float> in_h(n);
    for (int i = 0; i < n; ++i) {
        in_h[i] = 1.0f + static_cast<float>(i % 97) * 0.01f;
    }

    std::vector<float> ref_h(n);
    referenceHost(in_h.data(), ref_h.data(), n);

    float *in_d, *outDivergent_d, *outFree_d;
    CUDA_CHECK(cudaMalloc((void **)&in_d, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&outDivergent_d, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&outFree_d, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(in_d, in_h.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    dim3 dimBlock(blockSize);
    dim3 dimGrid(gridSize);

    // Warm-up launches (results discarded): the first time a kernel is
    // touched, the driver may need to JIT-compile PTX into SASS for the
    // actual GPU (this build targets -arch=sm_75; on a newer GPU the driver
    // JIT-compiles from the embedded PTX), which is a one-time cost that
    // has nothing to do with control divergence. Absorb it here so it
    // doesn't leak into whichever kernel is timed first below.
    divergentKernel<<<dimGrid, dimBlock>>>(in_d, outDivergent_d, n);
    CUDA_CHECK(cudaGetLastError());
    divergenceFreeKernel<<<dimGrid, dimBlock>>>(in_d, outFree_d, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;

    timer.start();
    divergentKernel<<<dimGrid, dimBlock>>>(in_d, outDivergent_d, n);
    CUDA_CHECK(cudaGetLastError());
    float msDivergent = timer.stopAndGetMs();

    timer.start();
    divergenceFreeKernel<<<dimGrid, dimBlock>>>(in_d, outFree_d, n);
    CUDA_CHECK(cudaGetLastError());
    float msFree = timer.stopAndGetMs();

    std::vector<float> outDivergent_h(n), outFree_h(n);
    CUDA_CHECK(cudaMemcpy(outDivergent_h.data(), outDivergent_d, n * sizeof(float),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(
        cudaMemcpy(outFree_h.data(), outFree_d, n * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(in_d));
    CUDA_CHECK(cudaFree(outDivergent_d));
    CUDA_CHECK(cudaFree(outFree_d));

    bool ok = true;
    for (int i = 0; i < n; ++i) {
        if (!nearlyEqual(outDivergent_h[i], ref_h[i]) || !nearlyEqual(outFree_h[i], ref_h[i])) {
            ok = false;
            fprintf(stderr,
                    "Mismatch at %d: divergent=%f divergence-free=%f ref=%f\n", i,
                    outDivergent_h[i], outFree_h[i], ref_h[i]);
            break;
        }
    }

    printf("n=%d, ITERS=%d, dimBlock=(%d), dimGrid=(%d)\n", n, ITERS, blockSize, gridSize);
    printf("GPU divergentKernel      time: %.3f ms\n", msDivergent);
    printf("GPU divergenceFreeKernel time: %.3f ms\n", msFree);
    printf("(Both kernels must agree with the CPU reference for PASS; which kernel\n"
           " is faster is reported for information, not used to decide PASS/FAIL.)\n");
    printf("%s\n", ok ? "PASS" : "FAIL");

    return ok ? 0 : 1;
}
