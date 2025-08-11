

#include "cuda_runtime.h"
#include <cmath>

#include "PolynomialSampling.cuh"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include "utils.cuh"

__constant__ SamplingRange d_range_const;
__constant__ Polynomial d_poly_const;

__global__ void addKernel(double* output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int i = tid; i < d_range_const.count; i += total_threads) {
        sample_polynomial_index(output, i);
    }
}



__host__ __device__
double Polynomial::evaluate(double x) const {
    double result = 0.0f;
    double xi = 1.0;
    for (int i = 0; i <= degree; ++i) {
        result += coeffs[i] * xi;
        xi *= x;
    }
    return result;
}

__host__ __device__
double SamplingRange::get_x(int i) const {
    return start + i * step;
}

inline __host__ __device__
void sample_polynomial_index(double* output, int i) {
    double x = d_range_const.get_x(i);
    output[i] = d_poly_const.evaluate(x);
}

//Note: output not freed here so it can be used without another allocation
void run_polynomial_sampling(const Polynomial& h_poly_in, const SamplingRange& h_range_in, double* h_output) {
    std::cout << "Starting Polynomial Sampling on GPU..." << std::endl;

    // Device pointer for output
    double* d_output = nullptr;

    // Allocate device memory for output
    CUDA_CHECK(cudaMalloc(&d_output, h_range_in.count * sizeof(double)));

    //Copy input structs to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(d_poly_const, &h_poly_in, sizeof(Polynomial)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_range_const, &h_range_in, sizeof(SamplingRange)));

    // Launch kernel
    int threadsPerBlock = 512;
    int blocks = (h_range_in.count + threadsPerBlock - 1) / threadsPerBlock;
    addKernel << <blocks, threadsPerBlock >> > (d_output);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, h_range_in.count * sizeof(double), cudaMemcpyDeviceToHost));

    std::cout << "Polynomial Sampling complete." << std::endl;

}
