#pragma once

#include "PolynomialSampling.cuh"
#include "device_launch_parameters.h"
#include <vector>

extern __constant__ SamplingRange d_range_const;
extern __constant__ Polynomial d_poly_const;


// Structure to hold the data for each potential inflection point
struct __align__(16) InflectionCandidate {
    int index;
    float change_magnitude;
    int pad0;
};
static_assert(sizeof(InflectionCandidate) % 16 == 0, "Candidate size should be multiple of 16");



// Pass 1: A grid-stride loop kernel to count the total number of potential inflection points.
__global__ void countPotentialInflections(const float* y, int* count, int N, int stencil_width);

// Pass 2: A grid-stride loop kernel to write the indices and magnitudes of the potential inflection points.
__global__ void writeInflectionCandidates(const float* __restrict__ y,
    InflectionCandidate* candidates,
    int* count,
    int N,
    int stencil_width);

__global__ void countPotentialInflections_shared(const float* y_global, int* count, int N, int stencil_width);
__global__ void writeInflectionCandidates_shared(const float* __restrict__ y_global,
    InflectionCandidate* candidates,
    int* count,
    int N,
    int stencil_width);

// The host-side function to run the entire process.
std::vector<int> run_find_inflections(const SamplingRange& h_range_in, float* d_output, int expected_inflections);