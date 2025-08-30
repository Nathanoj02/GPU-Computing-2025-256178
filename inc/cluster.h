#ifndef CLUSTER_H
#define CLUSTER_H

#include <stdint.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <float.h>  // For FLT_MAX
#include <string.h>

#include "kmeans.h"

/**
 * K-means clustering algorithm
 * 
 * @param params K-means parameters
 * @param distance_func Pointer to the distance function to use
 * @param minkowski_parameter Parameter for Minkowski distance (ignored for other distances)
 */
void k_means (
    KMeansParams *params,
    float (*distance_func)(const uint8_t*, const uint8_t*, unsigned int, float),
    float minkowski_parameter
);

#endif // CLUSTER_H