#include "PolynomialSampling.cuh"
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#pragma once
void runExperiment(const Polynomial& h_poly, const SamplingRange& h_range);