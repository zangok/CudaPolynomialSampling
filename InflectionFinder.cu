#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <vector>
#include "PolynomialSampling.cuh"
#include "utils.cuh"
#include "InflectionFinder.cuh"
#include <algorithm>
#include <cfloat> 

// Approximate second derivative divided by step^2
__device__ inline float second_derivative_approx(const float* y, int idx, float step) {
    return (y[idx - 1] - 2.0f * y[idx] + y[idx + 1]) / (step * step);
}

// Pass 1: count potential inflection points (sign change in second derivative)
__global__ void countPotentialInflections(const float* y, int* count, int N, float step) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int idx = tid + 1; idx < N - 2; idx += stride) {
        float d2y_i = second_derivative_approx(y, idx, step);
        float d2y_i1 = second_derivative_approx(y, idx + 1, step);

        if ((d2y_i <= 0.0f && d2y_i1 >= 0.0f) || (d2y_i >= 0.0f && d2y_i1 <= 0.0f)) {
            atomicAdd(count, 1);
        }
    }
}

// Pass 2: write inflection candidates with step passed and fixed indexing
__global__ void writeInflectionCandidates(const float* __restrict__ y,
    InflectionCandidate* candidates,
    int* count,
    int N,
    float step) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int idx = tid + 1; idx < N - 2; idx += stride) {
        float d2y_i = second_derivative_approx(y, idx, step);
        float d2y_i1 = second_derivative_approx(y, idx + 1, step);

        if ((d2y_i <= 0.0f && d2y_i1 >= 0.0f) || (d2y_i >= 0.0f && d2y_i1 <= 0.0f)) {
            int pos = atomicAdd(count, 1);
            float change_magnitude = fmaxf(fabsf(d2y_i), fabsf(d2y_i1));

            candidates[pos].index = idx;
            candidates[pos].change_magnitude = change_magnitude;
            candidates[pos].pad0 = 0;
        }
    }
}

// Host function to run the entire inflection finder pipeline
std::vector<int> run_find_inflections(const SamplingRange& h_range_in, float* d_output, int expected_inflections) {
    const int N = h_range_in.count;
    if (N < 3) return {};

    float step = h_range_in.step; // step size

    // Allocate and zero device counter
    int* d_count;
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(int)));

    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (blocks == 0) blocks = 1;

    // Count potential inflections
    countPotentialInflections << <blocks, threadsPerBlock >> > (d_output, d_count, N, step);
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
    writeInflectionCandidates << <blocks, threadsPerBlock >> > (d_output, d_candidates, d_count, N, step);
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
