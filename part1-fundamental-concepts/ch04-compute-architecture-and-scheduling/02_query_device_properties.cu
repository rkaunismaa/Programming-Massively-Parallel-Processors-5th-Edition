// Chapter 4: Compute architecture and scheduling
// §4.8  Querying device properties
//
// §4.8 introduces cudaGetDeviceCount() to find how many CUDA-capable
// devices are present:
//   int devCount;
//   cudaGetDeviceCount(&devCount);
// and cudaGetDeviceProperties() to fill in a cudaDeviceProp struct for
// each device index 0..devCount-1:
//   cudaDeviceProp devProp;
//   for (unsigned int i = 0; i < devCount; i++) {
//       cudaGetDeviceProperties(&devProp, i);
//       // Decide if device has sufficient resources/capabilities
//   }
//
// The section then walks through the specific devProp fields most
// relevant to reasoning about execution-resource assignment, which this
// sample prints for every visible device:
//   - devProp.maxThreadsPerBlock  -- max threads allowed in a block
//   - devProp.multiProcessorCount -- number of SMs
//   - devProp.clockRate           -- clock frequency (kHz); combined with
//                                    SM count, indicates peak throughput
//   - devProp.maxThreadsDim[0..2] -- max threads per block, per dimension
//   - devProp.maxGridSize[0..2]   -- max blocks per grid, per dimension
//   - devProp.regsPerBlock        -- registers available for a block
//   - devProp.warpSize            -- threads per warp (§4.4/§4.5)
//
// A few additional identifying fields not named in §4.8's text (device
// name, compute capability major/minor, total global memory, shared
// memory per block) are also printed since they are cheap, standard
// context for identifying which physical GPU is being described; they
// are called out separately from the book-listing fields below.
//
// PASS means: cudaGetDeviceCount() succeeds, the device count is > 0, and
// cudaGetDeviceProperties() succeeds for every device found.

#include <cstdio>

#include "../../common/cuda_utils.h"

int main() {
    int devCount = 0;
    cudaError_t countStatus = cudaGetDeviceCount(&devCount);

    bool ok = (countStatus == cudaSuccess) && (devCount > 0);

    if (countStatus != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(countStatus));
    } else {
        printf("cudaGetDeviceCount: %d CUDA-capable device(s) found\n\n", devCount);
    }

    for (int i = 0; ok && i < devCount; ++i) {
        cudaDeviceProp devProp;
        cudaError_t propStatus = cudaGetDeviceProperties(&devProp, i);
        if (propStatus != cudaSuccess) {
            fprintf(stderr, "cudaGetDeviceProperties failed for device %d: %s\n", i,
                    cudaGetErrorString(propStatus));
            ok = false;
            break;
        }

        printf("Device %d: %s\n", i, devProp.name);
        printf("  compute capability            : %d.%d\n", devProp.major, devProp.minor);

        // --- §4.8 listing fields ---
        printf("  maxThreadsPerBlock             : %d\n", devProp.maxThreadsPerBlock);
        printf("  multiProcessorCount (SMs)      : %d\n", devProp.multiProcessorCount);
        printf("  clockRate (kHz)                : %d\n", devProp.clockRate);
        printf("  maxThreadsDim[0,1,2]           : (%d, %d, %d)\n", devProp.maxThreadsDim[0],
               devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);
        printf("  maxGridSize[0,1,2]             : (%d, %d, %d)\n", devProp.maxGridSize[0],
               devProp.maxGridSize[1], devProp.maxGridSize[2]);
        printf("  regsPerBlock                   : %d\n", devProp.regsPerBlock);
        printf("  warpSize                       : %d\n", devProp.warpSize);

        // --- additional context, not named in §4.8's text ---
        printf("  totalGlobalMem (bytes)         : %zu\n", devProp.totalGlobalMem);
        printf("  sharedMemPerBlock (bytes)      : %zu\n", devProp.sharedMemPerBlock);
        printf("\n");
    }

    printf("%s\n", ok ? "PASS" : "FAIL");

    return ok ? 0 : 1;
}
