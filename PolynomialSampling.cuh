#pragma once

#include "cuda_runtime.h"
#include <cmath>
constexpr auto MAX_DEGREE = 8;
// Polynomial struct
struct Polynomial {
    double coeffs[MAX_DEGREE];
    int degree;

    __host__ __device__
        double evaluate(double x) const;
};

// SamplingRange struct
// start: starting x value (lowest)
// step: size of each step
// count: the amount of steps total for the range
struct SamplingRange {
    double start;
    double step;
    int count;

    __host__ __device__
        double get_x(int i) const;
};


extern __constant__ Polynomial d_poly_const;
extern __constant__ SamplingRange d_range_const;

// Function to sample a polynomial for index i, outputs in output
__host__ __device__ inline void sample_polynomial_index(double* output, int i);

__global__ void addKernel(double* output);

void run_polynomial_sampling(const Polynomial& h_poly_in, const SamplingRange& h_range_in, double* h_output);