/**
 * Distance metric functions for multi-dimensional data points.
 * Each function computes the distance between two points `a` and `b`
 * in a space defined by `dimensions`
 */

#ifndef DISTANCES_H
#define DISTANCES_H

#include <stdint.h>
#include <float.h>
#include <math.h>

/**
 * Squared Euclidean Distance
 * @note Returns squared distance to avoid sqrt for efficiency
 * @param a First data point
 * @param b Second data point
 * @param dimensions Number of dimensions
 * @param dummy Unused parameter to match function signature
 * @return Squared Euclidean distance
 */
static inline float squared_euclidean_distance(const uint8_t* a, const uint8_t* b, unsigned int dimensions, float dummy) {
    float sum = 0.0f;
    for (unsigned int i = 0; i < dimensions; i++) {
        float diff = (float)a[i] - (float)b[i];
        sum += diff * diff;
    }
    return sum;
}

/**
 * Euclidean Distance
 * @param a First data point
 * @param b Second data point
 * @param dimensions Number of dimensions
 * @param dummy Unused parameter to match function signature
 * @return Euclidean distance
 */
static inline float euclidean_distance(const uint8_t* a, const uint8_t* b, unsigned int dimensions, float dummy) {
    return sqrt(squared_euclidean_distance(a, b, dimensions, dummy));
}

/**
 * Manhattan Distance
 * @param a First data point
 * @param b Second data point
 * @param dimensions Number of dimensions
 * @param dummy Unused parameter to match function signature
 * @return Manhattan distance
 */
static inline float manhattan_distance(const uint8_t* a, const uint8_t* b, unsigned int dimensions, float dummy) {
    float sum = 0.0f;
    for (unsigned int i = 0; i < dimensions; i++) {
        float diff = (float)a[i] - (float)b[i];
        sum += fabs(diff);
    }
    return sum;
}

/**
 * Chebyshev Distance
 * @param a First data point
 * @param b Second data point
 * @param dimensions Number of dimensions
 * @param dummy Unused parameter to match function signature
 * @return Chebyshev distance
 */
static inline float chebyshev_distance(const uint8_t* a, const uint8_t* b, unsigned int dimensions, float dummy) {
    float max_diff = 0.0f;
    for (unsigned int i = 0; i < dimensions; i++) {
        float diff = fabs((float)a[i] - (float)b[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

/**
 * Minkowski Distance
 * @param a First data point
 * @param b Second data point
 * @param dimensions Number of dimensions
 * @param p Order of the norm (e.g., 2 for Euclidean, 1 for Manhattan)
 * @return Minkowski distance
 */
static inline float minkowski_distance(const uint8_t* a, const uint8_t* b, unsigned int dimensions, float p) {
    float sum = 0.0f;
    for (unsigned int i = 0; i < dimensions; i++) {
        float diff = fabs((float)a[i] - (float)b[i]);
        sum += pow(diff, p);
    }
    return pow(sum, 1.0f / p);
}

/**
 * Cosine Distance
 * @param a First data point
 * @param b Second data point
 * @param dimensions Number of dimensions
 * @param dummy Unused parameter to match function signature
 * @return Cosine distance
 */
static inline float cosine_distance(const uint8_t* a, const uint8_t* b, unsigned int dimensions, float dummy) {
    float dot_product = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (unsigned int i = 0; i < dimensions; i++) {
        float val_a = (float)a[i];
        float val_b = (float)b[i];
        dot_product += val_a * val_b;
        norm_a += val_a * val_a;
        norm_b += val_b * val_b;
    }

    if (norm_a == 0.0f || norm_b == 0.0f) {
        return 1.0f; // Max distance if one vector is zero
    }

    return 1.0f - (dot_product / (sqrt(norm_a) * sqrt(norm_b)));
}

#endif // DISTANCES_H