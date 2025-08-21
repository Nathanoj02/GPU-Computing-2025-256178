#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#include "data.h"
#include "lsh_cuda.cuh"
#include "error.cuh"

#define NAME_LEN 256

// Convert Point to Point_GPU
Point_GPU* convert_to_gpu_points(Point *points, size_t num_points) {
    Point_GPU *gpu_points = (Point_GPU*)malloc(num_points * sizeof(Point_GPU));
    
    for (size_t i = 0; i < num_points; i++) {
        gpu_points[i].coords = points[i].coords;
        gpu_points[i].id = points[i].id;
    }
    
    return gpu_points;
}

// Simple naive implementation for comparison (CPU only)
int naive_nearest_neighbors_cpu(Point_GPU *points, size_t num_points, Point_GPU *query_point, 
                               double max_distance, int *neighbors, double *distances, int dimensions) {
    int neighbor_count = 0;
    
    for (size_t i = 0; i < num_points; i++) {
        if (points[i].id == query_point->id) continue;  // Skip the query point itself
        
        double dist = calculate_distance_gpu(&points[i], query_point, dimensions);
        
        if (dist <= max_distance) {
            neighbors[neighbor_count] = i;
            distances[neighbor_count] = dist;
            neighbor_count++;
        }
    }
    
    return neighbor_count;
}

int main(int argc, char *argv[]) {
    // Get dimensions, L and k from command line arguments
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <dimensions> <L> <k>\n", argv[0]);
        return 1;
    }
    
    size_t dimensions = atoi(argv[1]);
    size_t L = atoi(argv[2]);
    size_t k = atoi(argv[3]);

    // Validate dimensions and number of hyperplanes
    if (dimensions <= 0 || L <= 0 || k <= 0) {
        fprintf(stderr, "Dimensions, L and k must be positive integers.\n");
        return 1;
    }

    // Check CUDA device
    int device_count;
    SAFE_CALL(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        return 1;
    }
    
    printf("Found %d CUDA device(s)\n", device_count);

    // Set CUDA device
    SAFE_CALL(cudaSetDevice(0));
    
    cudaDeviceProp prop;
    SAFE_CALL(cudaGetDeviceProperties(&prop, 0));
    printf("Using GPU: %s\n", prop.name);

    // Check if file exists
    char filename[NAME_LEN];
    snprintf(filename, sizeof(filename), "dataset/points_%zud.txt", dimensions);

    size_t num_points;
    Point *points = read_data(&num_points, filename);

    if (!points) {
        fprintf(stderr, "Error reading data from file: %s\n", filename);
        return 1;
    }

    printf("Loaded %zu points with %zu dimensions\n", num_points, dimensions);
    
    // Convert to GPU points
    Point_GPU *gpu_points = convert_to_gpu_points(points, num_points);

    // Initialize LSH for GPU
    LSH_GPU *lsh = init_lsh_gpu(dimensions, L, k, gpu_points, num_points);
    if (!lsh) {
        fprintf(stderr, "Failed to initialize LSH on GPU\n");
        return 1;
    }

    printf("Initialized LSH with L=%zu tables, k=%zu hyperplanes per table\n", L, k);

    // Build index
    printf("Building LSH index on GPU...\n");
    build_index_gpu(lsh);
    printf("Index built successfully\n");

    // Take first point as query point
    Point_GPU query_point;
    query_point.coords = gpu_points[0].coords;
    query_point.id = gpu_points[0].id;

    int *lsh_neighbors = (int*)malloc(num_points * sizeof(int));
    double *lsh_distances = (double*)malloc(num_points * sizeof(double));

    // Time LSH GPU algorithm
    cudaEvent_t start, stop;
    SAFE_CALL(cudaEventCreate(&start));
    SAFE_CALL(cudaEventCreate(&stop));

    SAFE_CALL(cudaEventRecord(start));
    int lsh_count = lsh_nearest_neighbors_gpu(lsh, &query_point, 0.2, lsh_neighbors, lsh_distances);
    SAFE_CALL(cudaEventRecord(stop));
    SAFE_CALL(cudaEventSynchronize(stop));

    float lsh_time_ms;
    SAFE_CALL(cudaEventElapsedTime(&lsh_time_ms, start, stop));

    // Naive nearest neighbors for comparison (CPU)
    int *naive_neighbors = (int*)malloc(num_points * sizeof(int));
    double *naive_distances = (double*)malloc(num_points * sizeof(double));
    
    clock_t cpu_start = clock();
    int naive_count = naive_nearest_neighbors_cpu(gpu_points, num_points, &query_point, 0.2, 
                                                naive_neighbors, naive_distances, dimensions);
    clock_t cpu_end = clock();

    double naive_time_ms = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC * 1000.0;

    // Print results
    printf("\n=== Results ===\n");
    printf("LSH GPU found %d neighbors in %.3f ms\n", lsh_count, lsh_time_ms);
    printf("Naive CPU found %d neighbors in %.3f ms\n", naive_count, naive_time_ms);
    printf("Speedup: %.2fx\n", naive_time_ms / lsh_time_ms);
    
    // Calculate recall (what fraction of true neighbors did LSH find)
    int matches = 0;
    for (int i = 0; i < lsh_count; i++) {
        for (int j = 0; j < naive_count; j++) {
            if (lsh_neighbors[i] == naive_neighbors[j]) {
                matches++;
                break;
            }
        }
    }
    
    double recall = naive_count > 0 ? (double)matches / naive_count : 0.0;
    printf("Recall: %.2f%% (%d/%d)\n", recall * 100, matches, naive_count);

    // Print some example neighbors
    printf("\nFirst 5 LSH neighbors:\n");
    for (int i = 0; i < lsh_count && i < 5; i++) {
        printf("  Point %d (distance: %.6f)\n", lsh_neighbors[i], lsh_distances[i]);
    }

    // Cleanup
    cleanup_lsh_gpu(lsh);

    for (size_t i = 0; i < num_points; i++) {
        free(points[i].coords);
    }
    free(points);
    free(gpu_points);
    
    free(lsh_neighbors);
    free(lsh_distances);
    free(naive_neighbors);
    free(naive_distances);
    
    SAFE_CALL(cudaEventDestroy(start));
    SAFE_CALL(cudaEventDestroy(stop));
    
    SAFE_CALL(cudaDeviceReset());
    
    printf("Program completed successfully\n");
    return 0;
}