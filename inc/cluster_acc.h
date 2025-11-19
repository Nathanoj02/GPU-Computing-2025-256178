#ifndef CLUSTER_ACC_H
#define CLUSTER_ACC_H

#include <stdint.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <float.h>  // For FLT_MAX
#include <string.h>

#include "kmeans.h"

#define TILE_SIZE 32

/**
 * K-means clustering algorithm with OpenACC
 * 
 * @param params KMeansParams structure containing all parameters
 */
void k_means_acc (KMeansParams* params);

/**
 * K-means++ clustering algorithm with OpenACC
 * 
 * @param params KMeansParams structure containing all parameters
 */
void k_means_pp_acc (KMeansParams* params);

/**
 * K-means clustering algorithm with OpenACC with pixel-based centroids
 * 
 * @param params KMeansParams structure containing all parameters
 */
void k_means_pixel_centroid (KMeansParams* params);

/**
 * K-means clustering algorithm with OpenACC with custom prototypes - ideal for working within a defined calibration
 * 
 * @param params KMeansParams structure containing all parameters
 * @param prototypes Initial prototypes for the clusters
 * @param calibration_mode If true, initialize prototypes randomly
 */
void k_means_custom_centroids (KMeansParams* params, uint8_t* prototypes, bool calibration_mode);

/**
 * K-means clustering algorithm with OpenACC with tiling
 * 
 * @param params KMeansParams structure containing all parameters
 */
void k_means_acc_tiled (KMeansParams* params);

/**
 * K-means clustering algorithm with OpenACC checking convergence every few iterations
 * 
 * @param params KMeansParams structure containing all parameters
 * @param check_convergence_step Step interval for checking convergence
 */
void k_means_acc_check_conv (KMeansParams* params, int check_convergence_step);

/**
 * K-means++ clustering algorithm with OpenACC with 2-step reduction (instead of atomic operations)
 * 
 * @param params KMeansParams structure containing all parameters
 */
void k_means_pp_acc_reduce (KMeansParams* params);

/**
 * K-means clustering algorithm with OpenACC (original version before fixes)
 * 
 * @param params KMeansParams structure containing all parameters
 */
void k_means_acc_old (KMeansParams* params);

#endif // CLUSTER_ACC_H