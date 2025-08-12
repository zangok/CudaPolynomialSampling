#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <vector>
#include "PolynomialSampling.cuh"
#include "utils.cuh"
#include "InflectionFinder.cuh"
#include <algorithm>
#include <cfloat> 

// stencil to handle floating point from close values 
__device__ inline float second_derivative_numerator(const float* y, int idx, int stencil_width) {

    float y_im = y[idx - stencil_width];
    float y_i = y[idx];
    float y_ip = y[idx + stencil_width];

    return fmaf(-2.0f, y_i, y_im + y_ip);
}

// Pass 1: count potential inflection points (sign change in second derivative)
__global__ void countPotentialInflections(const float* y, int* count, int N, int stencil_width) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Adjust loop bounds to avoid reading out of bounds with the wider stencil
    for (int idx = tid + stencil_width; idx < N - (stencil_width + 1); idx += stride) {
        float num_i = second_derivative_numerator(y, idx, stencil_width);
        float num_i1 = second_derivative_numerator(y, idx + 1, stencil_width);

        if ((num_i <= 0.0f && num_i1 >= 0.0f) || (num_i >= 0.0f && num_i1 <= 0.0f)) {
            atomicAdd(count, 1);
        }
    }
}

// Pass 2: write inflection candidates
__global__ void writeInflectionCandidates(const float* __restrict__ y,
    InflectionCandidate* candidates,
    int* count,
    int N,
    int stencil_width) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Adjust loop bounds here as well
    for (int idx = tid + stencil_width; idx < N - (stencil_width + 1); idx += stride) {
        float num_i = second_derivative_numerator(y, idx, stencil_width);
        float num_i1 = second_derivative_numerator(y, idx + 1, stencil_width);

        if ((num_i <= 0.0f && num_i1 >= 0.0f) || (num_i >= 0.0f && num_i1 <= 0.0f)) {
            int pos = atomicAdd(count, 1);
            float change_magnitude = fmaxf(fabsf(num_i), fabsf(num_i1));

            candidates[pos].index = idx;
            candidates[pos].change_magnitude = change_magnitude;
            candidates[pos].pad0 = 0;
        }
    }
}

// Host function to run the entire inflection finder pipeline
//Stencil is to deal with floating point errors
//Using floats for signficant speedup for cuda after analyzing with Nvidia Nsight
std::vector<int> run_find_inflections(const SamplingRange& h_range_in, float* d_output, int expected_inflections) {
    const int N = h_range_in.count;
    if (N < 3) return {};

    const float target_stable_step = 0.02f;

    // 2. Get the actual step size for the current high-density input.
    const float actual_step = h_range_in.step;

    // 3. Calculate the dynamic stencil width.
    //    This determines how many indices we need to step over to cover
    //    a physically stable distance.
    int stencil_width = static_cast<int>(roundf(target_stable_step / actual_step));

    // 4. Ensure the stencil width is at least 1.
    if (stencil_width < 1) {
        stencil_width = 1;
    }

    // Ensure N is large enough for the chosen stencil
    if (N < 2 * stencil_width + 2) return {};

    // Allocate and zero device counter
    int* d_count;
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

    int minGridSize;
    int blockSize;

    // This call suggests an optimal block size and the minimum grid size to achieve full occupancy
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, countPotentialInflections, 0, N));

    // threadsPerBlock is now the suggested blockSize
    int threadsPerBlock = blockSize;

    // Now calculate the number of blocks based on the data size and the optimal block size
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Round the number of blocks up to the minGridSize to avoid a partial wave
    blocks = (blocks + minGridSize - 1) / minGridSize * minGridSize;

    if (blocks == 0) blocks = 1;

    // Count potential inflections
    countPotentialInflections << <blocks, threadsPerBlock >> > (d_output, d_count, N, stencil_width);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back count
    int h_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_count == 0) {
        cudaFree(d_count);
        return {};
    }

    // Allocate candidate array on device
    InflectionCandidate* d_candidates;
    CUDA_CHECK(cudaMalloc(&d_candidates, h_count * sizeof(InflectionCandidate)));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int))); // reset counter for writing

    // Write candidates
    writeInflectionCandidates << <blocks, threadsPerBlock >> > (d_output, d_candidates, d_count, N, stencil_width);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy candidates back to host
    std::vector<InflectionCandidate> h_candidates(h_count);
    CUDA_CHECK(cudaMemcpy(h_candidates.data(), d_candidates, h_count * sizeof(InflectionCandidate), cudaMemcpyDeviceToHost));

    // Cleanup
    cudaFree(d_candidates);
    cudaFree(d_count);

    // Sort candidates by magnitude ascending (smallest magnitude = closest to zero)
    std::sort(h_candidates.begin(), h_candidates.end(), [](const InflectionCandidate& a, const InflectionCandidate& b) {
        return a.change_magnitude < b.change_magnitude;
        });

    // Extract top expected_inflections indices
    std::vector<int> h_indices;
    int num_results = std::min((int)h_candidates.size(), expected_inflections);
    for (int i = 0; i < num_results; ++i) {
        h_indices.push_back(h_candidates[i].index);
    }

    return h_indices;
}
