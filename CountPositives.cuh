#pragma once 


__global__ void runPositivesKernel(int* count, const double* __restrict__ y, int N);

int calc_positives(double* y, const SamplingRange& h_range_in);