
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>


/*
Idea: All weights abd bias in global mem
Divide n samples equally on GPU threads.
keep array of values on device for reduced transfers
This minimizes transfers.
*/

#include "PolynomialSampling.cuh" 
#include "utils.cuh"
#include "ExperimentManager.cuh"


int main() {
    // 1. Define and initialize host data for the experiment
    double h_coeffs[] = {1.0, 0.5, -1.0, 0.15};
    int degree = 3;
    int sample_count = 10000000;

    Polynomial h_poly{};
    for (int i = 0; i <= degree; ++i) {
        h_poly.coeffs[i] = h_coeffs[i];
    }
    h_poly.degree = degree;

    SamplingRange h_range = { -10.0, 20.0/sample_count, sample_count +1};

    // 2. Call the simple experiment runner function
    runExperiment(h_poly, h_range);

    std::cin.get();

    return 0;
}