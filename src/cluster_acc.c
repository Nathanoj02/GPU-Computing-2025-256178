#include "cluster_acc.h"
#include "distances.h"

void k_means_acc (
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
                        float distance = squared_euclidean_distance(
                            &img[i * img_width * dimensions + j * dimensions], 
                            &prototypes[p * dimensions], 
                            dimensions
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