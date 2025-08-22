#ifndef CLUSTER_HPP
#define CLUSTER_HPP

#include <stdint.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <float.h>  // For FLT_MAX
#include <string.h>

/**
 * K-means clustering algorithm
 * 
 * @param dst Destination image
 * @param img Source image
 * @param img_height Source image height
 * @param img_width Source image width
 * @param k Number of clusters
 * @param dimensions Number of dimensions (e.g., 3 for RGB, 1 for grayscale)
 * @param stab_error Error bound to reach to end the algorithm
 * @param max_iterations Maximum number of iterations
 */
void k_means (
    uint8_t* dst, uint8_t* img,
    size_t img_height, size_t img_width,
    unsigned int k, unsigned int dimensions,
    float stab_error, int max_iterations
);

#endif // CLUSTER_HPP