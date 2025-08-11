#pragma once 


__global__ void runPositivesKernel(int* count, const float* __restrict__ y, int N);

int calc_positives(float* y, const SamplingRange& h_range_in);