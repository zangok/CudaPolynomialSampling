#include "PolynomialSampling.cuh"
#include <iostream>
#include <chrono>
#include "InflectionFinder.cuh"
#include "utils.cuh"
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include "CountPositives.cuh"
#include <iomanip>
#include "InflectionFinderInterpolation.cuh"
#include <algorithm>
// CPU version of the sampling
static void run_polynomial_sampling_cpu(const Polynomial& poly, const SamplingRange& range, float* output) {
    for (int i = 0; i < range.count; ++i) {
        float x = range.get_x(i);
        output[i] = poly.evaluate(x);
    }
}
// CPU version of positive amount finder
static int run_calc_positive_cpu(float* y, const SamplingRange& h_range_in) {
    int count = 0;
    for (int i = 0; i < h_range_in.count; i++) {
        if (y[i] > 0.0)
            count++;
    }
    return count;
}

std::vector<int> run_find_inflections_cpu(const SamplingRange& h_range_in, const std::vector<float>& y_data, int expected_inflections) {
    const int N = h_range_in.count;
    if (N < 4 || expected_inflections <= 0) return {};

    // Vector to store all potential inflection candidates.
    std::vector<InflectionCandidate> h_candidates;

    // Iterate through the data to find all potential inflection points.
    for (int idx = 0; idx < N - 3; ++idx) {
        float slope1 = y_data[idx + 1] - y_data[idx];
        float slope2 = y_data[idx + 2] - y_data[idx + 1];
        float change_in_slope = slope2 - slope1;
        float next_change_in_slope = (y_data[idx + 3] - y_data[idx + 2]) - slope2;

        // Check for a sign change in the change of slope.
        if ((change_in_slope < -FLT_EPSILON && next_change_in_slope > FLT_EPSILON) ||
            (change_in_slope > FLT_EPSILON && next_change_in_slope < -FLT_EPSILON)) {

            // Add a new candidate to the vector.
            h_candidates.push_back({ idx + 1, std::fabs(change_in_slope) });
        }
    }

    if (h_candidates.empty() || h_candidates.size() < expected_inflections) {
        return {};
    }

    // Sort the candidates by magnitude in descending order.
    std::sort(h_candidates.begin(), h_candidates.end(), [](const InflectionCandidate& a, const InflectionCandidate& b) {
        return a.change_magnitude > b.change_magnitude;
        });

    // Select the top 'expected_inflections' and return their indices.
    std::vector<int> h_indices;
    h_indices.reserve(expected_inflections);
    for (int i = 0; i < expected_inflections; ++i) {
        h_indices.push_back(h_candidates[i].index);
    }

    return h_indices;
}
// Helper function to time and print the duration of a given task.
template<typename Func>
std::chrono::duration<double> timeAndPrint(const std::string& description, Func task) {
    auto t_start = std::chrono::high_resolution_clock::now();
    task();
    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = t_end - t_start;
    return duration;
}

void runExperiment(const Polynomial& h_poly, const SamplingRange& h_range) {
    std::cout << "--- Running Experiment for: " << h_range.count-1 << " values -- - \n";

    int count = h_range.count;

    // Host buffer for GPU output
    float* h_output_gpu = new float[count];

    // Device buffer for polynomial sampling output
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_output, count * sizeof(float)));

    std::chrono::duration<double> sampling_duration;
    std::chrono::duration<double> inflection_duration;
    std::chrono::duration<double> positives_duration;

    std::chrono::duration<double> sampling_duration_cpu;
    std::chrono::duration<double> inflection_duration_cpu;
    std::chrono::duration<double> positives_duration_cpu;

    std::vector<int> inflection_locations;
    int positives = 0;
    int positives_cpu = 0;

    // --- Time Polynomial Sampling ---
    sampling_duration = timeAndPrint("Polynomial Sampling", [&]() {
        run_polynomial_sampling(h_poly, h_range, d_output);
        CUDA_CHECK(cudaDeviceSynchronize());
        });

    // --- Time Inflection Finding ---
    int expected_inflections = h_poly.degree > 1 ? h_poly.degree - 2 : 0;

    inflection_duration = timeAndPrint("Inflection Finding", [&]() {
        inflection_locations = run_find_inflections(h_range, d_output, expected_inflections);
        CUDA_CHECK(cudaDeviceSynchronize());
        });

    // Copy sampled polynomial output from device to host for verification / further processing
    CUDA_CHECK(cudaMemcpy(h_output_gpu, d_output, count * sizeof(float), cudaMemcpyDeviceToHost));

    // --- Time Positive Count ---
    positives_duration = timeAndPrint("Positive Count", [&]() {
        positives = calc_positives(d_output, h_range);
        CUDA_CHECK(cudaDeviceSynchronize());
        });

    // --- Generate Text-Based Graph ---
    std::cout << "--- Text Graph ---\n";
    std::vector<int> graph_values;
    graph_values.reserve(21);
    int total_points = h_range.count;

    for (int i = 0; i < 21; i++) {
        int idx = static_cast<int>((static_cast<double>(i) / (20)) * (total_points - 1));
        std::cout << "#";
        graph_values.push_back(std::round(h_output_gpu[idx]));
        std::cout << std::round(h_output_gpu[idx]) << "\n";
    }

    for (int i = 5; i >= -5; i--) {
        for (int j = 0; j < 21; j++) {
            if (graph_values[j] == i) {
                std::cout << "#";
            }
            else {
                std::cout << "*";
            }
        }
        std::cout << "\n";
    }
    // --- Final Results Summary ---
    std::cout << "--- Experiment Results ---\n";
    std::cout << std::fixed << std::setprecision(2);

    std::cout << "Inflection Points:\n";
    for (int j : inflection_locations) {
        std::cout << "  (" << h_range.get_x(j) << ", " << h_output_gpu[j] << ")\n";
    }

    std::cout << "\nNumber of positive samples: " << positives << "\n";

    std::cout << "Runtime for sample generation: " << sampling_duration.count() << " seconds\n";
    std::cout << "Runtime for inflection point search: " << inflection_duration.count() << " seconds\n";
    std::cout << "Runtime for positive sample count: " << positives_duration.count() << " seconds\n";


    // --- CPU Timings ---

    // Buffer for CPU polynomial sampling output
    float* h_output_cpu = new float[count];

    sampling_duration_cpu = timeAndPrint("CPU Polynomial Sampling", [&]() {
        run_polynomial_sampling_cpu(h_poly, h_range, h_output_cpu);
        });

    inflection_duration_cpu = timeAndPrint("CPU Inflection Finding", [&]() {
        std::vector<float> y_data(h_output_cpu, h_output_cpu + count);
        auto inflections_cpu = run_find_inflections_cpu(h_range, y_data, expected_inflections);
        });

    positives_duration_cpu = timeAndPrint("CPU Positive Count", [&]() {
        positives_cpu = run_calc_positive_cpu(h_output_cpu, h_range);
        });
    // --- Print Summary ---

    std::cout << "\n--- Performance Comparison ---\n";
    std::cout << positives_cpu << "\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Polynomial Sampling: GPU = " << sampling_duration.count() << " s, CPU = " << sampling_duration_cpu.count() << " s\n";
    std::cout << "Inflection Finding:  GPU = " << inflection_duration.count() << " s, CPU = " << inflection_duration_cpu.count() << " s\n";
    std::cout << "Positive Count:      GPU = " << positives_duration.count() << " s, CPU = " << positives_duration_cpu.count() << " s\n";
    
    // Clean up host and device memory
    delete[] h_output_gpu;
    cudaFree(d_output);
}