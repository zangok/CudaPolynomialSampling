#pragma once


#include "PolynomialSampling.cuh"
#include "utils.cuh"

float find_best_inflection_onepass(const Polynomial& h_poly, const SamplingRange& h_range);

__global__ void kernel_find_best_inflection(
    float dx,                 // spacing (range.step can be used if equal)
    float abs_floor,          // absolute floor threshold
    float rel_factor,         // relative factor for tol
    float* best_score_out,    // device float* initially set to big number
    float* best_pos_out       // device float* to write fractional index (i + t)
);

__device__ void atomicMinFloatIfLess(float* addr, float val, float* out_old);
