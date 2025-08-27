#include "cluster_acc.h"
#include "distances.h"

void _k_means_acc (
    uint8_t* dst, uint8_t* img,
    size_t img_height, size_t img_width,
    unsigned int k, unsigned int dimensions,
    uint8_t* prototypes,
    float stab_error, int max_iterations,
    float minkowski_parameter
);


void k_means_acc(
    uint8_t* dst, uint8_t* img,
    size_t img_height, size_t img_width,
    unsigned int k, unsigned int dimensions,
    float stab_error, int max_iterations,
    float minkowski_parameter)
{
    srand(0);   // Seed for reproducibility

    // Create k prototypes with random values
    uint8_t* prototypes = (uint8_t*) malloc (sizeof(uint8_t) * k * dimensions);
    for (unsigned int i = 0; i < k * dimensions; i++) 
    {
        prototypes[i] = rand() % 256;
    }

    _k_means_acc(dst, img, img_height, img_width, k, dimensions, prototypes, stab_error, max_iterations, minkowski_parameter);

    // Free memory
    free (prototypes);
}

void k_means_pp_acc (
    uint8_t* dst, uint8_t* img,
    size_t img_height, size_t img_width,
    unsigned int k, unsigned int dimensions,
    float stab_error, int max_iterations,
    float minkowski_parameter)
{
    srand(0);   // Seed for reproducibility

    // Create k prototypes with random values
    uint8_t* prototypes = (uint8_t*) malloc (sizeof(uint8_t) * k * dimensions);

    size_t total_pixels = img_height * img_width;

    // Random 1st centroid
    for (unsigned int i = 0; i < dimensions; i++) 
    {
        prototypes[i] = rand() % 256;
    }

    // K-means++ initialization for remaining centroids
    float* distances = (float*) malloc(sizeof(float) * total_pixels);
    
    for (unsigned int centroid_idx = 1; centroid_idx < k; centroid_idx++)
    {
        // Calculate minimum distance from each pixel to existing centroids
        for (size_t pixel_idx = 0; pixel_idx < total_pixels; pixel_idx++)
        {
            float min_dist = FLT_MAX;
            
            // Find minimum distance to any existing centroid
            for (unsigned int existing_centroid = 0; existing_centroid < centroid_idx; existing_centroid++)
            {
                float dist = euclidean_distance(
                    img + pixel_idx * dimensions,
                    prototypes + existing_centroid * dimensions,
                    dimensions,
                    0
                );
                
                if (dist < min_dist)
                {
                    min_dist = dist;
                }
            }
            
            distances[pixel_idx] = min_dist;
        }
        
        // Find pixel with maximum minimum distance
        float max_min_dist = -1.0f;
        size_t next_centroid_pixel = 0;
        
        for (size_t pixel_idx = 0; pixel_idx < total_pixels; pixel_idx++)
        {
            if (distances[pixel_idx] > max_min_dist)
            {
                max_min_dist = distances[pixel_idx];
                next_centroid_pixel = pixel_idx;
            }
        }
        
        // Set the new centroid
        for (unsigned int dim = 0; dim < dimensions; dim++)
        {
            prototypes[centroid_idx * dimensions + dim] = 
                img[next_centroid_pixel * dimensions + dim];
        }
    }
    
    // Free temporary memory
    free(distances);

    // Call k-means
    _k_means_acc(dst, img, img_height, img_width, k, dimensions, prototypes, stab_error, max_iterations, minkowski_parameter);

    // Free memory
    free (prototypes);
}

void k_means_pixel_centroid (
    uint8_t* dst, uint8_t* img,
    size_t img_height, size_t img_width,
    unsigned int k, unsigned int dimensions,
    float stab_error, int max_iterations,
    float minkowski_parameter)
{
    srand(0);
    uint8_t* prototypes = (uint8_t*) malloc (sizeof(uint8_t) * k * dimensions);

    size_t total_pixels = img_height * img_width;

    // Select from actual image pixels
    for (unsigned int j = 0; j < k; j++)
    {
        size_t first_pixel_idx = rand() % total_pixels;
        for (unsigned int i = 0; i < dimensions; i++) 
        {
            prototypes[j * dimensions + i] = img[first_pixel_idx * dimensions + i];
        }
    }

    _k_means_acc(dst, img, img_height, img_width, k, dimensions, prototypes, stab_error, max_iterations, minkowski_parameter);

    free(prototypes);
}

void k_means_custom_centroids (
    uint8_t* dst, uint8_t* img,
    size_t img_height, size_t img_width,
    unsigned int k, unsigned int dimensions,
    uint8_t* prototypes, bool calibration_mode,
    float stab_error, int max_iterations,
    float minkowski_parameter)
{
    if (calibration_mode) {
        for (unsigned int i = 0; i < k * dimensions; i++) {
            prototypes[i] = rand() % 256;
        }
    }
    
    // Just call k-means, prototypes are handled externally
    _k_means_acc(dst, img, img_height, img_width, k, dimensions, prototypes, stab_error, max_iterations, minkowski_parameter);
}

void _k_means_acc (
    uint8_t* dst, uint8_t* img,
    size_t img_height, size_t img_width,
    unsigned int k, unsigned int dimensions,
    uint8_t* prototypes,
    float stab_error, int max_iterations,
    float minkowski_parameter)
{   
    uint8_t* assigned_img = (uint8_t*) calloc (img_height * img_width, sizeof(uint8_t));
    uint8_t* old_prototypes = (uint8_t*) malloc (sizeof(uint8_t) * k * dimensions);

    // Use local arrays for better GPU memory access patterns
    size_t total_pixels = img_height * img_width;
    
    bool bound_reached = false;

    // Single data region for the entire algorithm
    #pragma acc data copyin(img[:img_height*img_width*dimensions]) \
                copy(prototypes[:k*dimensions]) \
                create(assigned_img[:total_pixels], old_prototypes[:k*dimensions]) \
                copyout(dst[:img_height*img_width*dimensions])
    {
        for (int iteration_count = 0; 
             !bound_reached && iteration_count < max_iterations; 
             iteration_count++) 
        {
            // Save old prototypes
            #pragma acc parallel loop present(prototypes, old_prototypes)
            for (unsigned int i = 0; i < k * dimensions; i++) {
                old_prototypes[i] = prototypes[i];
            }

            // Pixel assignment phase - find nearest prototype for each pixel
            #pragma acc parallel loop collapse(2) present(img, prototypes, assigned_img) \
                        gang vector
            for (size_t i = 0; i < img_height; i++)
            {
                for (size_t j = 0; j < img_width; j++)
                {
                    float min_distance = FLT_MAX;
                    uint8_t assigned_prototype_index = 0;
                    size_t img_offset = i * img_width * dimensions + j * dimensions;

                    // Find nearest prototype
                    for (unsigned int p = 0; p < k; p++)
                    {
                        float distance = manhattan_distance(
                            &img[img_offset], 
                            &prototypes[p * dimensions], 
                            dimensions,
                            minkowski_parameter
                        );

                        if (distance < min_distance) {
                            min_distance = distance;
                            assigned_prototype_index = p;
                        }
                    }
                    assigned_img[i * img_width + j] = assigned_prototype_index;
                }
            }

            // Accumulation phase - sum up pixel values for each cluster
            // Use atomic operations for accumulation
            uint64_t* sums = (uint64_t*) calloc(k * dimensions, sizeof(uint64_t));
            uint64_t* counts = (uint64_t*) calloc(k, sizeof(uint64_t));
            
            #pragma acc data create(sums[:k*dimensions], counts[:k])
            {
                // Initialize arrays to zero on device
                #pragma acc parallel loop present(sums)
                for (unsigned int i = 0; i < k * dimensions; i++) sums[i] = 0;
                
                #pragma acc parallel loop present(counts)
                for (unsigned int i = 0; i < k; i++) counts[i] = 0;

                // Accumulate pixel values for each cluster
                #pragma acc parallel loop collapse(2) present(img, assigned_img, sums, counts) \
                            gang vector
                for (size_t i = 0; i < img_height; i++)
                {
                    for (size_t j = 0; j < img_width; j++)
                    {
                        uint8_t cluster_id = assigned_img[i * img_width + j];
                        size_t img_offset = i * img_width * dimensions + j * dimensions;
                        
                        // Atomic updates for thread safety
                        for (unsigned int d = 0; d < dimensions; d++) {
                            #pragma acc atomic update
                            sums[cluster_id * dimensions + d] += img[img_offset + d];
                        }
                        #pragma acc atomic update
                        counts[cluster_id]++;
                    }
                }

                // Update prototypes with new means
                #pragma acc parallel loop present(prototypes, sums, counts)
                for (unsigned int i = 0; i < k; i++)
                {
                    if (counts[i] != 0)
                    {
                        for (unsigned int d = 0; d < dimensions; d++) 
                        {
                            prototypes[i * dimensions + d] = (uint8_t)((float)sums[i * dimensions + d] / counts[i]);
                        }
                    }
                }
            }
            
            free(sums);
            free(counts);

            // Convergence check - calculate maximum change in prototypes
            float max_diff_squared = 0.0f;
            #pragma acc parallel loop present(prototypes, old_prototypes) \
                        reduction(max:max_diff_squared)
            for (unsigned int i = 0; i < k; i++)
            {
                // Use distance from distance.h
                float distance_squared = squared_euclidean_distance(
                    &prototypes[i * dimensions],
                    &old_prototypes[i * dimensions],
                    dimensions,
                    0.0f
                );

                if (distance_squared > max_diff_squared) {
                    max_diff_squared = distance_squared;
                }
            }
            
            bound_reached = (max_diff_squared <= stab_error);
        }

        // Final assignment - vectorized
        #pragma acc parallel loop collapse(2) present(dst, assigned_img, prototypes) \
                    gang vector
        for (size_t i = 0; i < img_height; i++)
        {
            for (size_t j = 0; j < img_width; j++)
            {
                uint8_t index = assigned_img[i * img_width + j];
                size_t dst_offset = i * img_width * dimensions + j * dimensions;
                
                for (unsigned int d = 0; d < dimensions; d++) 
                {
                    dst[dst_offset + d] = prototypes[index * dimensions + d];
                }
            }
        }
    }
    
    // Free memory
    free(assigned_img);
    free(old_prototypes);
}


// Original version before fixes
void k_means_acc_old (
    uint8_t* dst, uint8_t* img,
    size_t img_height, size_t img_width,
    unsigned int k, unsigned int dimensions,
    float stab_error, int max_iterations,
    float minkowski_parameter)
{
    srand(0);   // Seed for reproducibility

    // Create k prototypes with random values
    uint8_t* prototypes = (uint8_t*) malloc (sizeof(uint8_t) * k * dimensions);
    for (unsigned int i = 0; i < k * dimensions; i++) 
    {
        prototypes[i] = rand() % 256;
    }
    
    uint8_t* assigned_img = (uint8_t*) calloc (img_height * img_width, sizeof(uint8_t));  // Map : pixels -> cluster number

    // Array for calculating means
    uint64_t* sums = (uint64_t*) malloc (sizeof(uint64_t) * k * dimensions);
    uint64_t* counts = (uint64_t*) malloc (sizeof(uint64_t) * k);

    uint8_t* old_prototypes = (uint8_t*) malloc (sizeof(uint8_t) * k * dimensions);

    bool bound_reached = false;

    // Loop until prototypes are stable
    // FIXED: Added assigned_img to data clause and changed copy to copyout for dst
    #pragma acc data copy(prototypes[:k*dimensions]) \
                copyin(img[:img_height*img_width*dimensions]) \
                create(assigned_img[:img_height*img_width], sums[:k*dimensions], counts[:k], old_prototypes[:k*dimensions]) \
                copyout(dst[:img_height*img_width*dimensions])
    {
        for (
            int iteration_count = 0; 
            !bound_reached && iteration_count < max_iterations; 
            iteration_count++) 
        {
            // FIXED: Move memcpy inside data region or use acc update
            #pragma acc update host(prototypes[:k*dimensions])
            memcpy(old_prototypes, prototypes, k * dimensions * sizeof(uint8_t));
            #pragma acc update device(old_prototypes[:k*dimensions])

            // Resetting sums and counts on GPU
            #pragma acc parallel loop
            for (unsigned int i = 0; i < k * dimensions; i++) sums[i] = 0;

            #pragma acc parallel loop
            for (unsigned int i = 0; i < k; i++) counts[i] = 0;

            // Associate each pixel to nearest prototype (with Euclidian distance)
            // FIXED: Removed reduction clause - reductions on array sections with computed indices are problematic
            #pragma acc parallel loop collapse(2)
            for (size_t i = 0; i < img_height; i++)
            {
                for (size_t j = 0; j < img_width; j++)
                {
                    float min_distance = FLT_MAX;
                    int assigned_prototype_index = -1;

                    for (unsigned int p = 0; p < k; p++)
                    {
                        float distance = manhattan_distance(
                            &img[i * img_width * dimensions + j * dimensions], 
                            &prototypes[p * dimensions], 
                            dimensions,
                            minkowski_parameter
                        );

                        if (distance < min_distance) {
                            min_distance = distance;
                            assigned_prototype_index = p;
                        }
                    }
                    assigned_img[i * img_width + j] = assigned_prototype_index;
                }
            }

            // FIXED: Separate loop for accumulating sums and counts using atomic operations
            #pragma acc parallel loop collapse(2)
            for (size_t i = 0; i < img_height; i++)
            {
                for (size_t j = 0; j < img_width; j++)
                {
                    int cluster_id = assigned_img[i * img_width + j];
                    
                    // Update sums and counts with atomic operations
                    for (unsigned int d = 0; d < dimensions; d++) {
                        #pragma acc atomic update
                        sums[cluster_id * dimensions + d] += img[i * img_width * dimensions + j * dimensions + d];
                    }
                    #pragma acc atomic update
                    counts[cluster_id]++;
                }
            }

            // Update values of the prototypes to the means of the associated pixels
            #pragma acc parallel loop
            for (unsigned int i = 0; i < k; i++)
            {
                if (counts[i] != 0)
                {
                    for (unsigned int d = 0; d < dimensions; d++) 
                    {
                        prototypes[i * dimensions + d] = (uint8_t) ((float) sums[i * dimensions + d] / counts[i]);
                    }
                }
            }

            // FIXED: Move convergence check to host (requires host data)
            #pragma acc update host(prototypes[:k*dimensions])
            
            // Calculate differences
            bound_reached = true;

            for (unsigned int i = 0; i < k; i++)
            {
                float distance_squared = squared_euclidean_distance(
                    &prototypes[i * dimensions],
                    &old_prototypes[i * dimensions],
                    dimensions,
                    0.0f
                );

                if (distance_squared > stab_error)
                {
                    bound_reached = false;
                    break;
                }
            }
            
            // FIXED: Update device with new prototypes if continuing
            if (!bound_reached) {
                #pragma acc update device(prototypes[:k*dimensions])
            }
        }

        // Substitute each pixel with the corresponding prototype value
        // FIXED: Removed separate data region since dst is already in the outer data region
        #pragma acc parallel loop collapse(2)
        for (size_t i = 0; i < img_height; i++)
        {
            for (size_t j = 0; j < img_width; j++)
            {
                int index = assigned_img[i * img_width + j];
    
                for (unsigned int d = 0; d < dimensions; d++) 
                {
                    dst[i * img_width * dimensions + j * dimensions + d] = prototypes[index * dimensions + d];
                }
            }
        }
    }
    
    // Free memory
    free (prototypes);
    free (assigned_img);
    free (sums);
    free (counts);
    free (old_prototypes);
}