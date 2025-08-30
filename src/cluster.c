#include "cluster.h"
#include "distances.h"
#include "kmeans.h"

void _k_means (
    KMeansParams *params,
    uint8_t* prototypes,
    float (*distance_func)(const uint8_t*, const uint8_t*, unsigned int, float),
    float minkowski_parameter
);


void k_means(
    KMeansParams *params,
    float (*distance_func)(const uint8_t*, const uint8_t*, unsigned int, float),
    float minkowski_parameter)
{
    srand(0);   // Seed for reproducibility

    // Create k prototypes with random values
    uint8_t* prototypes = (uint8_t*) malloc (sizeof(uint8_t) * params->k * params->dimensions);
    for (unsigned int i = 0; i < params->k * params->dimensions; i++) 
    {
        prototypes[i] = rand() % 256;
    }

    _k_means(params, prototypes, distance_func, minkowski_parameter);

    // Free memory
    free (prototypes);
}

void _k_means (
    KMeansParams *params,
    uint8_t* prototypes,
    float (*distance_func)(const uint8_t*, const uint8_t*, unsigned int, float),
    float minkowski_parameter)
{
    uint8_t* assigned_img = (uint8_t*) calloc (params->img_height * params->img_width, sizeof(uint8_t));  // Map : pixels -> cluster number

    // Array for calculating means
    uint64_t* sums = (uint64_t*) malloc (sizeof(uint64_t) * params->k * params->dimensions);
    uint64_t* counts = (uint64_t*) malloc (sizeof(uint64_t) * params->k);

    uint8_t* old_prototypes = (uint8_t*) malloc (sizeof(uint8_t) * params->k * params->dimensions);

    bool bound_reached = false;

    // Loop until prototypes are stable
    for (
        int iteration_count = 0; 
        !bound_reached && iteration_count < params->max_iterations; 
        iteration_count++) 
    {
        memcpy(old_prototypes, prototypes, params->k * params->dimensions * sizeof(uint8_t));    // Save old values for calculating differences

        // Resetting sums and counts
        for (unsigned int i = 0; i < params->k * params->dimensions; i++) sums[i] = 0;
        for (unsigned int i = 0; i < params->k; i++) counts[i] = 0;

        // Associate each pixel to nearest prototype (with Euclidian distance)
        for (size_t i = 0; i < params->img_height; i++)
        {
            for (size_t j = 0; j < params->img_width; j++)
            {
                float min_distance = FLT_MAX;
                int assigned_prototype_index = -1;

                for (unsigned int p = 0; p < params->k; p++)
                {
                    float distance = distance_func(
                        &params->img[i * params->img_width * params->dimensions + j * params->dimensions], 
                        &prototypes[p * params->dimensions], 
                        params->dimensions,
                        minkowski_parameter
                    );

                    if (distance < min_distance) {
                        min_distance = distance;
                        assigned_prototype_index = p;
                    }
                }
                assigned_img[i * params->img_width + j] = assigned_prototype_index;

                // Update sums and counts
                for (unsigned int d = 0; d < params->dimensions; d++) {
                    sums[assigned_prototype_index * params->dimensions + d] += params->img[i * params->img_width * params->dimensions + j * params->dimensions + d];
                }
                counts[assigned_prototype_index]++;
            }
        }

        // Update values of the prototypes to the means of the associated pixels
        for (unsigned int i = 0; i < params->k; i++)
        {
            if (counts[i] != 0)
            {
                for (unsigned int d = 0; d < params->dimensions; d++) 
                {
                    prototypes[i * params->dimensions + d] = (uint8_t) ((float) sums[i * params->dimensions + d] / counts[i]);
                }
            }
        }

        // Calculate differences
        bound_reached = true;

        for (unsigned int i = 0; i < params->k; i++)
        {
            float distance_squared = squared_euclidean_distance(
                &prototypes[i * params->dimensions],
                &old_prototypes[i * params->dimensions],
                params->dimensions,
                0.0f
            );

            if (distance_squared > params->stab_error)
            {
                bound_reached = false;
                break;
            }
        }
    }

    // Substitute each pixel with the corresponding prototype value
    for (size_t i = 0; i < params->img_height; i++)
    {
        for (size_t j = 0; j < params->img_width; j++)
        {
            int index = assigned_img[i * params->img_width + j];

            for (unsigned int d = 0; d < params->dimensions; d++) 
            {
                params->dst[i * params->img_width * params->dimensions + j * params->dimensions + d] = prototypes[index * params->dimensions + d];
            }
        }
    }
    
    // Free memory
    free (assigned_img);
    free (sums);
    free (counts);
    free (old_prototypes);
}