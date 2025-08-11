// InflectionFinder_samples.cu
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include "PolynomialSampling.cuh"
#include "utils.cuh"            // CUDA_CHECK etc
#include "InflectionFinderInterpolation.cuh" // defines InflectionCandidate if you want; we'll use our own here
#include <cstdint>

extern __constant__ Polynomial d_poly_const;
extern __constant__ SamplingRange d_range_const;

__device__ void atomicMinFloatIfLess(float* addr, float val, float* out_old) {
    // works only for non-negative floats (our "score" will be >= 0)
    int* addr_int = reinterpret_cast<int*>(addr);
    int old_int = *addr_int;
    while (true) {
        float old_f = __int_as_float(old_int);
        if (!(val < old_f)) {
            if (out_old) *out_old = old_f;
            return; // no update needed
        }
        int new_int = __float_as_int(val);
        int prev = atomicCAS(addr_int, old_int, new_int);
        if (prev == old_int) {
            // we succeeded in installing new value
            if (out_old) *out_old = __int_as_float(prev);
            return;
        }
        // someone else updated meanwhile; retry with latest value
        old_int = prev;
    }
}

// Kernel: single-pass search, keep best (smallest score)
__global__ void kernel_find_best_inflection(
    float dx,                 // spacing (range.step can be used if equal)
    float abs_floor,          // absolute floor threshold
    float rel_factor,         // relative factor for tol
    float* best_score_out,    // device float* initially set to big number
    float* best_pos_out       // device float* to write fractional index (i + t)
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int N = d_range_const.count;

    for (int i = tid + 1; i <= N - 3; i += stride) { // we access i-1 .. i+2
        // compute y at i-1..i+2 via SamplingRange + Polynomial
        float x_im1 = d_range_const.get_x(i - 1);
        float x_i = d_range_const.get_x(i);
        float x_ip1 = d_range_const.get_x(i + 1);
        float x_ip2 = d_range_const.get_x(i + 2);

        double y_im1 = (double)d_poly_const.evaluate(x_im1);
        double y_i = (double)d_poly_const.evaluate(x_i);
        double y_ip1 = (double)d_poly_const.evaluate(x_ip1);
        double y_ip2 = (double)d_poly_const.evaluate(x_ip2);

        // central second differences s2(i) and s2(i+1)
        double s2 = y_ip1 - 2.0 * y_i + y_im1;
        double s2_next = y_ip2 - 2.0 * y_ip1 + y_i;
        if (dx != 1.0) {
            double dx2 = dx * dx;
            s2 /= dx2;
            s2_next /= dx2;
        }

        double mag = fmax(fabs(s2), fabs(s2_next));
        double tol = fmax((double)abs_floor, mag * (double)rel_factor);

        // require sign change (or zero) and magnitude above noise floor
        if ((s2 * s2_next < 0.0) && mag > tol)
        {
            // compute fractional root position between sample i and i+1
            double root_pos;
            if (fabs(s2) <= 1e-300) {
                root_pos = (double)i;
            }
            else if (fabs(s2_next) <= 1e-300) {
                root_pos = (double)(i + 1);
            }
            else {
                double t = fabs(s2) / (fabs(s2) + fabs(s2_next)); // fraction toward i+1
                root_pos = (double)i + t;
            }

            // compute score for ranking: we want smallest residual magnitude
            // use the smaller of the two magnitudes as a conservative indicator
            float score = (float)fmin(fabs(s2), fabs(s2_next));

            // attempt to update global best (if our score is smaller)
            float old_val;
            atomicMinFloatIfLess(best_score_out, score, &old_val);
            // if we succeeded (old_val > score), set position to ours
            if (old_val > score) {
                // store position; no strict atomic ordering needed because we just set after CAS
                // small race: another thread could set new best in between, but since we just
                // established that old_val > score and CAS succeeded, we currently hold the best score.
                *best_pos_out = (float)root_pos;
            }
        }
    }
}

float find_best_inflection_onepass(const Polynomial& h_poly, const SamplingRange& h_range) {
    // copy poly to device (assume you manage that)
    Polynomial d_poly;
    cudaMemcpyToSymbol(d_poly_const, &h_poly, sizeof(Polynomial));
    // optional: or allocate/copy explicitly

    // device outputs
    float* d_best_score;
    float* d_best_pos;
    CUDA_CHECK(cudaMalloc(&d_best_score, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_best_pos, sizeof(float)));

    float init_score = 1e30f;
    CUDA_CHECK(cudaMemcpy(d_best_score, &init_score, sizeof(float), cudaMemcpyHostToDevice));
    float init_pos = -1.0f;
    CUDA_CHECK(cudaMemcpy(d_best_pos, &init_pos, sizeof(float), cudaMemcpyHostToDevice));

    // launch
    const int tpb = 256;
    int blocks = (h_range.count + tpb - 1) / tpb;

    // choose dx: if samples uniformly spaced, use range.step; else pass explicit dx
    float dx = h_range.step;

    kernel_find_best_inflection << <blocks, tpb >> > (

        dx,
        /*abs_floor*/ 1e-8f,
        /*rel_factor*/ 1e-3f,
        d_best_score,
        d_best_pos
        );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // copy back best pos
    float h_best_pos;
    CUDA_CHECK(cudaMemcpy(&h_best_pos, d_best_pos, sizeof(float), cudaMemcpyDeviceToHost));

    // cleanup
    cudaFree(d_best_score);
    cudaFree(d_best_pos);

    return h_best_pos; // fractional index (i + t). round if you need integer index.
}