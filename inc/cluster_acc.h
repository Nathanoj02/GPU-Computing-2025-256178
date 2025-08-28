#ifndef CLUSTER_ACC_HPP
#define CLUSTER_ACC_HPP

#include <stdint.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <float.h>  // For FLT_MAX
#include <string.h>

#define TILE_SIZE 32

/**
 * K-means clustering algorithm with OpenACC
 * 
 * @param dst Destination image
 * @param img Source image
 * @param img_height Source image height
 * @param img_width Source image width
 * @param k Number of clusters
 * @param dimensions Number of dimensions (e.g., 3 for RGB, 1 for grayscale)
 * @param stab_error Error bound to reach to end the algorithm
 * @param max_iterations Maximum number of iterations
 * @param minkowski_parameter Parameter for Minkowski distance (ignored for other distances)
 */
void k_means_acc (
    uint8_t* dst, uint8_t* img,
    size_t img_height, size_t img_width,
    unsigned int k, unsigned int dimensions,
    float stab_error, int max_iterations,
    float minkowski_parameter
);

/**
 * K-means++ clustering algorithm with OpenACC
 * 
 * @param dst Destination image
 * @param img Source image
 * @param img_height Source image height
 * @param img_width Source image width
 * @param k Number of clusters
 * @param dimensions Number of dimensions (e.g., 3 for RGB, 1 for grayscale)
 * @param stab_error Error bound to reach to end the algorithm
 * @param max_iterations Maximum number of iterations
 * @param minkowski_parameter Parameter for Minkowski distance (ignored for other distances)
 */
void k_means_pp_acc (
    uint8_t* dst, uint8_t* img,
    size_t img_height, size_t img_width,
    unsigned int k, unsigned int dimensions,
    float stab_error, int max_iterations,
    float minkowski_parameter
);

/**
 * K-means clustering algorithm with OpenACC with pixel-based centroids
 * 
 * @param dst Destination image
 * @param img Source image
 * @param img_height Source image height
 * @param img_width Source image width
 * @param k Number of clusters
 * @param dimensions Number of dimensions (e.g., 3 for RGB, 1 for grayscale)
 * @param stab_error Error bound to reach to end the algorithm
 * @param max_iterations Maximum number of iterations
 * @param minkowski_parameter Parameter for Minkowski distance (ignored for other distances)
 */
void k_means_pixel_centroid (
    uint8_t* dst, uint8_t* img,
    size_t img_height, size_t img_width,
    unsigned int k, unsigned int dimensions,
    float stab_error, int max_iterations,
    float minkowski_parameter
);

/**
 * K-means clustering algorithm with OpenACC with custom prototypes - ideal for working within a defined calibration
 * 
 * @param dst Destination image
 * @param img Source image
 * @param img_height Source image height
 * @param img_width Source image width
 * @param k Number of clusters
 * @param dimensions Number of dimensions (e.g., 3 for RGB, 1 for grayscale)
 * @param prototypes Initial prototypes for the clusters
 * @param calibration_mode If true, initialize prototypes randomly
 * @param stab_error Error bound to reach to end the algorithm
 * @param max_iterations Maximum number of iterations
 * @param minkowski_parameter Parameter for Minkowski distance (ignored for other distances)
 */
void k_means_custom_centroids (
    uint8_t* dst, uint8_t* img,
    size_t img_height, size_t img_width,
    unsigned int k, unsigned int dimensions,
    uint8_t* prototypes, bool calibration_mode,
    float stab_error, int max_iterations,
    float minkowski_parameter
);

/**
 * K-means clustering algorithm with OpenACC with tiling
 * 
 * @param dst Destination image
 * @param img Source image
 * @param img_height Source image height
 * @param img_width Source image width
 * @param k Number of clusters
 * @param dimensions Number of dimensions (e.g., 3 for RGB, 1 for grayscale)
 * @param prototypes Initial prototypes for the clusters
 * @param calibration_mode If true, initialize prototypes randomly
 * @param stab_error Error bound to reach to end the algorithm
 * @param max_iterations Maximum number of iterations
 * @param minkowski_parameter Parameter for Minkowski distance (ignored for other distances)
 */
void k_means_acc_tiled (
    uint8_t* dst, uint8_t* img,
    size_t img_height, size_t img_width,
    unsigned int k, unsigned int dimensions,
    float stab_error, int max_iterations,
    float minkowski_parameter
);

/**
 * K-means clustering algorithm with OpenACC checking convergence every few iterations
 * 
 * @param dst Destination image
 * @param img Source image
 * @param img_height Source image height
 * @param img_width Source image width
 * @param k Number of clusters
 * @param dimensions Number of dimensions (e.g., 3 for RGB, 1 for grayscale)
 * @param stab_error Error bound to reach to end the algorithm
 * @param max_iterations Maximum number of iterations
 * @param minkowski_parameter Parameter for Minkowski distance (ignored for other distances)
 * @param check_convergence_step Step interval for checking convergence
 */
void k_means_acc_check_conv (
    uint8_t* dst, uint8_t* img,
    size_t img_height, size_t img_width,
    unsigned int k, unsigned int dimensions,
    float stab_error, int max_iterations,
    float minkowski_parameter, int check_convergence_step
);

/**
 * K-means clustering algorithm with OpenACC (original version before fixes)
 * 
 * @param dst Destination image
 * @param img Source image
 * @param img_height Source image height
 * @param img_width Source image width
 * @param k Number of clusters
 * @param dimensions Number of dimensions (e.g., 3 for RGB, 1 for grayscale)
 * @param stab_error Error bound to reach to end the algorithm
 * @param max_iterations Maximum number of iterations
 * @param minkowski_parameter Parameter for Minkowski distance (ignored for other distances)
 */
void k_means_acc_old (
    uint8_t* dst, uint8_t* img,
    size_t img_height, size_t img_width,
    unsigned int k, unsigned int dimensions,
    float stab_error, int max_iterations,
    float minkowski_parameter
);

#endif // CLUSTER_ACC_HPP