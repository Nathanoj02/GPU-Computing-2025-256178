#ifndef __LSH_CUDA_CUH__
#define __LSH_CUDA_CUH__

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "error.cuh"
#include "gpu_hash_table.cuh"

#define SEED 0
#define THREADS_PER_BLOCK 256
#define MAX_BUCKETS_PER_TABLE 65536  // 2^16

// Structure for a point in n-dimensional space (GPU version)
typedef struct {
    double *coords;
    int id;
} Point_GPU;

// Bucket structure for GPU
typedef struct {
    int *point_indices;     // Array of point indices in this bucket
    int count;              // Number of points in this bucket
    int capacity;           // Capacity of point_indices array
} Bucket_GPU;

// Hash table for GPU
typedef struct {
    Bucket_GPU *buckets;
    int num_buckets;
} HashTable_GPU;

// LSH structure for GPU
typedef struct {
    Point_GPU *points;          // Array of all points on GPU
    Point_GPU *h_points;        // Host copy of points
    size_t num_points;          // Number of points
    int dimensions;             // Dimensionality of points
    size_t num_hyperplanes;     // Number of hyperplanes
    double *hyperplanes;        // Flattened hyperplane normal vectors on GPU
    double *h_hyperplanes;      // Host copy of hyperplanes
    HashTable_GPU *hash_tables; // Hash tables on GPU
    HashTable_GPU *h_hash_tables; // Host copy of hash tables
    
    // GPU-specific arrays
    int *d_candidate_flags;     // Device candidate tracking
    int *d_candidates_buffer;   // Device candidate buffer
    int *d_candidate_count;     // Device candidate count
    
    int L;                      // Number of tables
    int k;                      // Hyperplanes per table
    
    // GPU memory management
    bool gpu_initialized;
} LSH_GPU;

// Function declarations
extern "C" {
    LSH_GPU* init_lsh_gpu(int dimensions, size_t L, size_t k, Point_GPU *points, size_t num_points);
    void build_index_gpu(LSH_GPU *lsh);
    int lsh_nearest_neighbors_gpu(LSH_GPU *lsh, Point_GPU *query_point, double max_distance, int *neighbors, double *distances);
    void cleanup_lsh_gpu(LSH_GPU *lsh);
    double calculate_distance_gpu(Point_GPU *p1, Point_GPU *p2, int dimensions);
}

// CUDA kernel declarations
__global__ void generate_hyperplanes_kernel(double *hyperplanes, int num_hyperplanes, int dimensions, unsigned long long seed);
__global__ void compute_hashes_kernel(Point_GPU *points, double *hyperplanes, int *hash_values, 
                                     size_t num_points, int L, int k, int dimensions);
__global__ void populate_buckets_kernel(LSH_GPU *lsh, int *hash_values);
__global__ void find_candidates_kernel(double *hyperplanes, HashTable_GPU *hash_tables, Point_GPU *query, 
                                     int *candidates, int *candidate_count, int *candidate_flags,
                                     int L, int k, int dimensions);
__global__ void calculate_distances_kernel(Point_GPU *points, Point_GPU *query, int *candidates, int candidate_count, 
                                         double max_distance, int *neighbors, double *distances, int *neighbor_count,
                                         int dimensions);

// Utility functions
__device__ int compute_table_hash_gpu(double *hyperplanes, Point_GPU *point, int table_idx, int k, int dimensions);
__device__ double calculate_distance_device(Point_GPU *p1, Point_GPU *p2, int dimensions);

#endif // __LSH_CUDA_CUH__