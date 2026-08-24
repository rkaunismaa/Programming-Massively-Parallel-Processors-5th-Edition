#ifndef PMPP_CUDA_UTILS_H
#define PMPP_CUDA_UTILS_H

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                         \
        if (err__ != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,\
                    cudaGetErrorString(err__));                             \
            exit(EXIT_FAILURE);                                            \
        }                                                                   \
    } while (0)

inline bool nearlyEqual(float a, float b, float eps = 1e-3f) {
    float diff = fabsf(a - b);
    float largest = fmaxf(fabsf(a), fabsf(b));
    return diff <= eps * fmaxf(1.0f, largest);
}

class GpuTimer {
public:
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start() { CUDA_CHECK(cudaEventRecord(start_)); }
    float stopAndGetMs() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

private:
    cudaEvent_t start_, stop_;
};

#endif  // PMPP_CUDA_UTILS_H
