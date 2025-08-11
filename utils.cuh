#pragma once

#include <iostream>


inline void checkCudaError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d - %s\n",
            file, line, cudaGetErrorString(err));
        std::cin.get(); // Pause for you to read the error
        exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) \
    do { \
        checkCudaError((call), __FILE__, __LINE__); \
        cudaError_t errSync = cudaDeviceSynchronize(); \
        checkCudaError(errSync, __FILE__, __LINE__); \
    } while (0)