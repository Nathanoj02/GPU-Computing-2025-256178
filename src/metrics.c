#include "metrics.h"

double elbow_method (
    uint8_t *src_img, uint8_t *res_img, 
    size_t img_height, size_t img_width,
    unsigned int k, unsigned int dimensions,
    float (*distance_func)(const uint8_t*, const uint8_t*, unsigned int, float),
    float minkowski_parameter)
{
    double error = 0.0;

    for (size_t i = 0; i < img_height * img_width; i++)
    {
        int diff = distance_func(
            &src_img[i * dimensions], 
            &res_img[i * dimensions], 
            dimensions, 
            minkowski_parameter
        );

        double pixel_error = diff * diff;

        error += pixel_error;
    }

    return error;
}


// Function to find which cluster each pixel belongs to based on res_img
void extract_cluster_assignments(uint8_t *res_img, uint8_t *src_img, uint8_t *assignments,
                                size_t img_height, size_t img_width, 
                                unsigned int k, unsigned int dimensions,
                                float (*distance_func)(const uint8_t*, const uint8_t*, unsigned int, float),
                                float minkowski_parameter)
{
    // First, extract unique cluster centers from res_img
    uint8_t *centroids = (uint8_t*) malloc(k * dimensions * sizeof(uint8_t));
    int found_centroids = 0;
    
    // Simple approach: find first k unique colors in res_img as centroids
    for (size_t i = 0; i < img_height && found_centroids < k; i++) {
        for (size_t j = 0; j < img_width && found_centroids < k; j++) {
            size_t pixel_offset = i * img_width * dimensions + j * dimensions;
            
            // Check if this color is already in our centroids
            bool is_new = true;
            for (int c = 0; c < found_centroids; c++) {
                bool same_color = true;
                for (unsigned int d = 0; d < dimensions; d++) {
                    if (centroids[c * dimensions + d] != res_img[pixel_offset + d]) {
                        same_color = false;
                        break;
                    }
                }
                if (same_color) {
                    is_new = false;
                    break;
                }
            }
            
            if (is_new) {
                for (unsigned int d = 0; d < dimensions; d++) {
                    centroids[found_centroids * dimensions + d] = res_img[pixel_offset + d];
                }
                found_centroids++;
            }
        }
    }
    
    // Now assign each pixel to its nearest centroid
    for (size_t i = 0; i < img_height; i++) {
        for (size_t j = 0; j < img_width; j++) {
            size_t pixel_offset = i * img_width * dimensions + j * dimensions;
            size_t assign_offset = i * img_width + j;
            
            float min_distance = FLT_MAX;
            uint8_t best_cluster = 0;
            
            for (int c = 0; c < found_centroids; c++) {
                float dist = distance_func(&src_img[pixel_offset], &centroids[c * dimensions], 
                                         dimensions, minkowski_parameter);
                if (dist < min_distance) {
                    min_distance = dist;
                    best_cluster = c;
                }
            }
            
            assignments[assign_offset] = best_cluster;
        }
    }
    
    free(centroids);
}


double silhouette_method (
    uint8_t *src_img, uint8_t *res_img, 
    size_t img_height, size_t img_width,
    unsigned int k, unsigned int dimensions,
    float (*distance_func)(const uint8_t*, const uint8_t*, unsigned int, float),
    float minkowski_parameter)
{
    size_t total_pixels = img_height * img_width;
    
    // Extract cluster assignments from the k-means result
    uint8_t *assignments = (uint8_t*) malloc(total_pixels * sizeof(uint8_t));
    extract_cluster_assignments(res_img, src_img, assignments, img_height, img_width, 
                               k, dimensions, distance_func, minkowski_parameter);
    
    // Count pixels in each cluster
    unsigned int *cluster_sizes = (unsigned int*) calloc(k, sizeof(unsigned int));
    for (size_t i = 0; i < total_pixels; i++) {
        cluster_sizes[assignments[i]]++;
    }
    
    // Calculate silhouette coefficient for each pixel
    double total_silhouette = 0.0;
    unsigned int valid_pixels = 0;
    
    for (size_t pixel_idx = 0; pixel_idx < total_pixels; pixel_idx++) {
        size_t i = pixel_idx / img_width;
        size_t j = pixel_idx % img_width;
        size_t pixel_offset = i * img_width * dimensions + j * dimensions;
        
        uint8_t own_cluster = assignments[pixel_idx];
        
        // Skip if cluster has only one point (silhouette undefined)
        if (cluster_sizes[own_cluster] <= 1) {
            continue;
        }
        
        // Calculate a(i): average distance to points in same cluster
        double sum_intra_distance = 0.0;
        unsigned int intra_count = 0;
        
        for (size_t other_idx = 0; other_idx < total_pixels; other_idx++) {
            if (other_idx != pixel_idx && assignments[other_idx] == own_cluster) {
                size_t other_i = other_idx / img_width;
                size_t other_j = other_idx % img_width;
                size_t other_offset = other_i * img_width * dimensions + other_j * dimensions;
                
                float dist = distance_func(&src_img[pixel_offset], &src_img[other_offset], 
                                         dimensions, minkowski_parameter);
                sum_intra_distance += dist;
                intra_count++;
            }
        }
        
        double a_i = (intra_count > 0) ? sum_intra_distance / intra_count : 0.0;
        
        // Calculate b(i): minimum average distance to points in other clusters
        double b_i = DBL_MAX;
        
        for (unsigned int other_cluster = 0; other_cluster < k; other_cluster++) {
            if (other_cluster == own_cluster || cluster_sizes[other_cluster] == 0) {
                continue;
            }
            
            double sum_inter_distance = 0.0;
            unsigned int inter_count = 0;
            
            for (size_t other_idx = 0; other_idx < total_pixels; other_idx++) {
                if (assignments[other_idx] == other_cluster) {
                    size_t other_i = other_idx / img_width;
                    size_t other_j = other_idx % img_width;
                    size_t other_offset = other_i * img_width * dimensions + other_j * dimensions;
                    
                    float dist = distance_func(&src_img[pixel_offset], &src_img[other_offset], 
                                             dimensions, minkowski_parameter);
                    sum_inter_distance += dist;
                    inter_count++;
                }
            }
            
            if (inter_count > 0) {
                double avg_inter_distance = sum_inter_distance / inter_count;
                if (avg_inter_distance < b_i) {
                    b_i = avg_inter_distance;
                }
            }
        }
        
        // Calculate silhouette coefficient for this pixel
        if (b_i != DBL_MAX && (a_i > 0.0 || b_i > 0.0)) {
            double s_i = (b_i - a_i) / fmax(a_i, b_i);
            total_silhouette += s_i;
            valid_pixels++;
        }
    }
    
    // Calculate average silhouette coefficient
    double avg_silhouette = (valid_pixels > 0) ? total_silhouette / valid_pixels : 0.0;
    
    // Cleanup
    free(assignments);
    free(cluster_sizes);
    
    return avg_silhouette;
}

// Optimized version with sampling for large images
double silhouette_method_sampled (
    uint8_t *src_img, uint8_t *res_img, 
    size_t img_height, size_t img_width,
    unsigned int k, unsigned int dimensions,
    float (*distance_func)(const uint8_t*, const uint8_t*, unsigned int, float),
    float minkowski_parameter,
    unsigned int max_samples)  // Maximum number of pixels to sample
{
    size_t total_pixels = img_height * img_width;
    
    // If image is small enough, use full calculation
    if (total_pixels <= max_samples) {
        return silhouette_method(src_img, res_img, img_height, img_width, k, 
                               dimensions, distance_func, minkowski_parameter);
    }
    
    // Extract cluster assignments
    uint8_t *assignments = (uint8_t*) malloc(total_pixels * sizeof(uint8_t));
    extract_cluster_assignments(res_img, src_img, assignments, img_height, img_width, 
                               k, dimensions, distance_func, minkowski_parameter);
    
    // Create sample of pixels (stratified sampling to ensure all clusters represented)
    unsigned int *cluster_sizes = (unsigned int*) calloc(k, sizeof(unsigned int));
    for (size_t i = 0; i < total_pixels; i++) {
        cluster_sizes[assignments[i]]++;
    }
    
    // Calculate sample size per cluster
    unsigned int *samples_per_cluster = (unsigned int*) malloc(k * sizeof(unsigned int));
    unsigned int total_samples = 0;
    
    for (unsigned int c = 0; c < k; c++) {
        if (cluster_sizes[c] > 0) {
            samples_per_cluster[c] = fmax(1, (max_samples * cluster_sizes[c]) / total_pixels);
            total_samples += samples_per_cluster[c];
        } else {
            samples_per_cluster[c] = 0;
        }
    }
    
    // Collect sample indices
    size_t *sample_indices = (size_t*) malloc(total_samples * sizeof(size_t));
    unsigned int sample_count = 0;
    
    srand(0); // Fixed seed for reproducibility
    
    for (unsigned int c = 0; c < k; c++) {
        unsigned int collected = 0;
        unsigned int attempts = 0;
        
        while (collected < samples_per_cluster[c] && attempts < cluster_sizes[c] * 2) {
            size_t random_idx = rand() % total_pixels;
            if (assignments[random_idx] == c) {
                sample_indices[sample_count++] = random_idx;
                collected++;
            }
            attempts++;
        }
    }
    
    // Calculate silhouette for sampled pixels
    double total_silhouette = 0.0;
    unsigned int valid_pixels = 0;
    
    for (unsigned int sample_i = 0; sample_i < sample_count; sample_i++) {
        size_t pixel_idx = sample_indices[sample_i];
        size_t i = pixel_idx / img_width;
        size_t j = pixel_idx % img_width;
        size_t pixel_offset = i * img_width * dimensions + j * dimensions;
        
        uint8_t own_cluster = assignments[pixel_idx];
        
        if (cluster_sizes[own_cluster] <= 1) {
            continue;
        }
        
        // Calculate a(i) using all pixels in same cluster
        double sum_intra_distance = 0.0;
        unsigned int intra_count = 0;
        
        for (size_t other_idx = 0; other_idx < total_pixels; other_idx++) {
            if (other_idx != pixel_idx && assignments[other_idx] == own_cluster) {
                size_t other_i = other_idx / img_width;
                size_t other_j = other_idx % img_width;
                size_t other_offset = other_i * img_width * dimensions + other_j * dimensions;
                
                float dist = distance_func(&src_img[pixel_offset], &src_img[other_offset], 
                                         dimensions, minkowski_parameter);
                sum_intra_distance += dist;
                intra_count++;
            }
        }
        
        double a_i = (intra_count > 0) ? sum_intra_distance / intra_count : 0.0;
        
        // Calculate b(i)
        double b_i = DBL_MAX;
        
        for (unsigned int other_cluster = 0; other_cluster < k; other_cluster++) {
            if (other_cluster == own_cluster || cluster_sizes[other_cluster] == 0) {
                continue;
            }
            
            double sum_inter_distance = 0.0;
            unsigned int inter_count = 0;
            
            for (size_t other_idx = 0; other_idx < total_pixels; other_idx++) {
                if (assignments[other_idx] == other_cluster) {
                    size_t other_i = other_idx / img_width;
                    size_t other_j = other_idx % img_width;
                    size_t other_offset = other_i * img_width * dimensions + other_j * dimensions;
                    
                    float dist = distance_func(&src_img[pixel_offset], &src_img[other_offset], 
                                             dimensions, minkowski_parameter);
                    sum_inter_distance += dist;
                    inter_count++;
                }
            }
            
            if (inter_count > 0) {
                double avg_inter_distance = sum_inter_distance / inter_count;
                if (avg_inter_distance < b_i) {
                    b_i = avg_inter_distance;
                }
            }
        }
        
        // Calculate silhouette coefficient
        if (b_i != DBL_MAX && (a_i > 0.0 || b_i > 0.0)) {
            double s_i = (b_i - a_i) / fmax(a_i, b_i);
            total_silhouette += s_i;
            valid_pixels++;
        }
    }
    
    double avg_silhouette = (valid_pixels > 0) ? total_silhouette / valid_pixels : 0.0;
    
    // Cleanup
    free(assignments);
    free(cluster_sizes);
    free(samples_per_cluster);
    free(sample_indices);
    
    return avg_silhouette;
}