#include "PolynomialSampling.cuh"
#include <iostream>
#include <chrono>
#include "CountPositives.cuh"
#include "utils.cuh"
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cmath>


//A atomic each thread does a count, may be unefficent.
//Another implementation I did think of was to use y as the storage for counting positives?
//I found that alot of the times, a atomic add was significantly faster than the reduction method below in
//Nvidia NSight
__global__ void runPositivesKernel(int* count, const double* __restrict__ y, int N) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Use a grid-stride loop to ensure all elements are checked
	for (int i = tid; i < N; i += gridDim.x * blockDim.x) {
		if (y[i] > 0) {
			atomicAdd(count, 1);
		}
	}
}

__global__ void runPositivesKernel_branchless(int* d_global_count, const double* __restrict__ y, int N) {
	// Shared memory array to store the counts for the current block
	extern __shared__ int s_block_count[];

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int local_count = 0;

	// Use a grid-stride loop to ensure all elements are checked
	for (int i = tid; i < N; i += gridDim.x * blockDim.x) {
		// This is the key change. The boolean expression (y[i] > 0)
		// evaluates to 1 or 0, which we can directly add without a conditional branch.
		local_count += (y[i] > 0);
	}

	// Store the local sum in shared memory
	s_block_count[threadIdx.x] = local_count;
	__syncthreads();

	
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (threadIdx.x < s) {
			s_block_count[threadIdx.x] += s_block_count[threadIdx.x + s];
		}
		__syncthreads();
	}

	// The first thread of the block atomically updates the global counter
	if (threadIdx.x == 0) {
		atomicAdd(d_global_count, s_block_count[0]);
	}
}

// Calculates positive count using the device array y
int calc_positives(double* d_y, const SamplingRange& h_range_in) {
	int N = h_range_in.count;

	if (N <= 0) return 0;

	int* d_count = nullptr;
	int h_count = 0; // Correctly initialize host variable
	CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
	CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

	//Launch config
	int threadsPerBlock = 256;
	int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

	// Pass the array size N d_y to the kernel for the grid-stride loop
	//runPositivesKernel << <blocksPerGrid, threadsPerBlock >> > (d_count, d_y, N);
	runPositivesKernel_branchless << <blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(int) >> > (d_count, d_y, N);

	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
	
	//copy the result from the device to the host
	CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));

	cudaFree(d_count);

	return h_count;
}
