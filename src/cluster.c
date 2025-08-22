#include "cluster.h"

void k_means (
    uint8_t* dst, uint8_t* img,
    size_t img_height, size_t img_width,
    unsigned int k, unsigned int dimensions,
    float stab_error, int max_iterations) 
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
    for (
        int iteration_count = 0; 
        !bound_reached && iteration_count < max_iterations; 
        iteration_count++) 
    {
        memcpy(old_prototypes, prototypes, k * dimensions * sizeof(uint8_t));    // Save old values for calculating differences

        // Resetting sums and counts
        for (unsigned int i = 0; i < k * dimensions; i++) sums[i] = 0;
        for (unsigned int i = 0; i < k; i++) counts[i] = 0;

        // Associate each pixel to nearest prototype (with Euclidian distance)
        for (size_t i = 0; i < img_height; i++)
        {
            for (size_t j = 0; j < img_width; j++)
            {
                float min_distance = FLT_MAX;
                int assigned_prototype_index = -1;

                for (unsigned int p = 0; p < k; p++)
                {
                    float distance = 0;

                    // Calculate Euclidian distance
                    for (unsigned int d = 0; d < dimensions; d++) {
                        float diff = img[i * img_width * dimensions + j * dimensions + d] - prototypes[p * dimensions + d];
                        distance += diff * diff;
                    }
                    distance = sqrt(distance);

                    if (distance < min_distance) {
                        min_distance = distance;
                        assigned_prototype_index = p;
                    }
                }
                assigned_img[i * img_width + j] = assigned_prototype_index;

                // Update sums and counts
                for (unsigned int d = 0; d < dimensions; d++) {
                    sums[assigned_prototype_index * dimensions + d] += img[i * img_width * dimensions + j * dimensions + d];
                }
                counts[assigned_prototype_index]++;
            }
        }

        // Update values of the prototypes to the means of the associated pixels
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

        // Calculate differences
        bound_reached = true;

        for (unsigned int i = 0; i < k; i++)
        {
            float distance_squared = 0;

            for (unsigned int d = 0; d < dimensions; d++) 
            {
                float diff = prototypes[i * dimensions + d] - old_prototypes[i * dimensions + d];
                distance_squared += diff * diff;
            }

            if (distance_squared > stab_error)
            {
                bound_reached = false;
                break;
            }
        }
    }

    // Substitute each pixel with the corresponding prototype value
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
    
    // Free memory
    free (prototypes);
    free (assigned_img);
    free (sums);
    free (counts);
    free (old_prototypes);
}