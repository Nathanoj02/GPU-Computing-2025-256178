#include <stdint.h>
#include <malloc.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>

/**
 * Elbow method for determining optimal number of clusters
 * @param src_img Original image data
 * @param res_img Image data after k-means
 * @param img_height Image height
 * @param img_width Image width
 * @param k Number of clusters
 * @param dimensions Number of color channels (e.g., 3 for RGB)
 * @param distance_func Pointer to distance function
 * @param minkowski_parameter Parameter for Minkowski distance (if applicable)
 * @return Total squared error
 */
double elbow_method (
    uint8_t *src_img, uint8_t *res_img, 
    size_t img_height, size_t img_width,
    unsigned int k, unsigned int dimensions,
    float (*distance_func)(const uint8_t*, const uint8_t*, unsigned int, float),
    float minkowski_parameter
);

/**
 * Sampled silhouette method for evaluating clustering quality
 * @param src_img Original image data
 * @param res_img Image data after k-means
 * @param img_height Image height
 * @param img_width Image width
 * @param k Number of clusters
 * @param dimensions Number of color channels (e.g., 3 for RGB)
 * @param distance_func Pointer to distance function
 * @param minkowski_parameter Parameter for Minkowski distance (if applicable)
 * @param max_samples Maximum number of pixels to sample for silhouette calculation
 * @return Average silhouette score
 */
double silhouette_method_sampled (
    uint8_t *src_img, uint8_t *res_img, 
    size_t img_height, size_t img_width,
    unsigned int k, unsigned int dimensions,
    float (*distance_func)(const uint8_t*, const uint8_t*, unsigned int, float),
    float minkowski_parameter,
    unsigned int max_samples);