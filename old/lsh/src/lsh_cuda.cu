#include "lsh_cuda.cuh"
#include <cmath>
#include <cstdlib>
#include <cstring>

// Device function to compute hash for a single table
__device__ int compute_table_hash_gpu(double *hyperplanes, Point_GPU *point, int table_idx, int k, int dimensions) {
    int hash = 0;
    int base = table_idx * k * dimensions;
    
    for (int h = 0; h < k; h++) {
        double dot = 0.0;
        int hp_base = base + h * dimensions;
        
        for (int d = 0; d < dimensions; d++) {
            dot += point->coords[d] * hyperplanes[hp_base + d];
        }
        hash = (hash << 1) | (dot >= 0 ? 1 : 0);
    }
    return hash;
}

// Device function to calculate distance between two points
__device__ double calculate_distance_device(Point_GPU *p1, Point_GPU *p2, int dimensions) {
    double sum = 0.0;
    for (int d = 0; d < dimensions; d++) {
        double diff = p1->coords[d] - p2->coords[d];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// Kernel to generate hyperplanes
__global__ void generate_hyperplanes_kernel(double *hyperplanes, int num_hyperplanes, int dimensions, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_hyperplanes * dimensions) return;
    
    int hp_idx = idx / dimensions;
    int dim_idx = idx % dimensions;
    
    // Initialize random state
    curandState state;
    curand_init(seed, idx, 0, &state);
    
    // Generate random value [-1, 1]
    hyperplanes[idx] = curand_uniform_double(&state) * 2.0 - 1.0;
    
    __syncthreads();
    
    // Normalize hyperplane (only first thread per hyperplane)
    if (dim_idx == 0) {
        double norm = 0.0;
        int base = hp_idx * dimensions;
        
        for (int d = 0; d < dimensions; d++) {
            norm += hyperplanes[base + d] * hyperplanes[base + d];
        }
        norm = sqrt(norm);
        
        if (norm > 0.0) {
            for (int d = 0; d < dimensions; d++) {
                hyperplanes[base + d] /= norm;
            }
        }
    }
}

// Kernel to compute hash values for all points and tables
__global__ void compute_hashes_kernel(Point_GPU *points, double *hyperplanes, int *hash_values, 
                                     size_t num_points, int L, int k, int dimensions) {
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_idx >= num_points) return;
    
    for (int table_idx = 0; table_idx < L; table_idx++) {
        int hash = compute_table_hash_gpu(hyperplanes, &points[point_idx], table_idx, k, dimensions);
        hash_values[point_idx * L + table_idx] = hash;
    }
}

// Kernel to find candidates for a query point
__global__ void find_candidates_kernel(double *hyperplanes, HashTable_GPU *hash_tables, Point_GPU *query, 
                                     int *candidates, int *candidate_count, int *candidate_flags,
                                     int L, int k, int dimensions) {
    int table_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (table_idx >= L) return;
    
    // Compute hash for query point in this table
    int hash_key = compute_table_hash_gpu(hyperplanes, query, table_idx, k, dimensions);
    hash_key = hash_key % hash_tables[table_idx].num_buckets;
    
    // Get bucket
    Bucket_GPU *bucket = &hash_tables[table_idx].buckets[hash_key];
    
    // Add all points in this bucket to candidates
    for (int i = 0; i < bucket->count; i++) {
        int point_idx = bucket->point_indices[i];
        
        // Use atomic compare-and-swap to avoid duplicates
        if (atomicCAS(&candidate_flags[point_idx], 0, 1) == 0) {
            int pos = atomicAdd(candidate_count, 1);
            candidates[pos] = point_idx;
        }
    }
}

// Kernel to calculate distances and filter neighbors
__global__ void calculate_distances_kernel(Point_GPU *points, Point_GPU *query, int *candidates, int candidate_count, 
                                         double max_distance, int *neighbors, double *distances, int *neighbor_count,
                                         int dimensions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= candidate_count) return;
    
    int point_idx = candidates[idx];
    Point_GPU *point = &points[point_idx];
    
    // Skip if it's the same point
    if (point->id == query->id) return;
    
    double dist = calculate_distance_device(query, point, dimensions);
    
    if (dist <= max_distance) {
        int pos = atomicAdd(neighbor_count, 1);
        neighbors[pos] = point_idx;
        distances[pos] = dist;
    }
}

// Initialize LSH structure for GPU
extern "C" LSH_GPU* init_lsh_gpu(int dimensions, size_t L, size_t k, Point_GPU *points, size_t num_points) {
    LSH_GPU *lsh = (LSH_GPU*)malloc(sizeof(LSH_GPU));
    lsh->L = L;
    lsh->k = k;
    lsh->dimensions = dimensions;
    lsh->num_points = num_points;
    lsh->num_hyperplanes = L * k;
    lsh->gpu_initialized = false;
    
    // Store host points
    lsh->h_points = points;
    
    // Allocate GPU memory for points
    SAFE_CALL(cudaMalloc(&lsh->points, num_points * sizeof(Point_GPU)));
    
    // Allocate and copy point coordinates to GPU
    for (size_t i = 0; i < num_points; i++) {
        double *d_coords;
        SAFE_CALL(cudaMalloc(&d_coords, dimensions * sizeof(double)));
        SAFE_CALL(cudaMemcpy(d_coords, points[i].coords, dimensions * sizeof(double), cudaMemcpyHostToDevice));
        
        Point_GPU temp_point = {d_coords, points[i].id};
        SAFE_CALL(cudaMemcpy(&lsh->points[i], &temp_point, sizeof(Point_GPU), cudaMemcpyHostToDevice));
    }
    
    // Allocate GPU memory for hyperplanes
    SAFE_CALL(cudaMalloc(&lsh->hyperplanes, lsh->num_hyperplanes * dimensions * sizeof(double)));
    
    // Allocate host memory for hyperplanes
    lsh->h_hyperplanes = (double*)malloc(lsh->num_hyperplanes * dimensions * sizeof(double));
    
    // Allocate GPU memory for hash tables
    SAFE_CALL(cudaMalloc(&lsh->hash_tables, L * sizeof(HashTable_GPU)));
    lsh->h_hash_tables = (HashTable_GPU*)malloc(L * sizeof(HashTable_GPU));
    
    // Initialize hash tables
    for (int i = 0; i < L; i++) {
        int num_buckets = 1 << k;
        if (num_buckets > MAX_BUCKETS_PER_TABLE) {
            num_buckets = MAX_BUCKETS_PER_TABLE;
        }
        
        lsh->h_hash_tables[i].num_buckets = num_buckets;
        
        // Allocate GPU memory for buckets
        SAFE_CALL(cudaMalloc(&lsh->h_hash_tables[i].buckets, num_buckets * sizeof(Bucket_GPU)));
        
        // Initialize empty buckets
        Bucket_GPU *temp_buckets = (Bucket_GPU*)malloc(num_buckets * sizeof(Bucket_GPU));
        for (int j = 0; j < num_buckets; j++) {
            int expected_capacity = (num_points / num_buckets) * 1.5; // 50% extra space
            if (expected_capacity < 10) expected_capacity = 10;
            
            temp_buckets[j].capacity = expected_capacity;
            temp_buckets[j].count = 0;
            SAFE_CALL(cudaMalloc(&temp_buckets[j].point_indices, expected_capacity * sizeof(int)));
        }
        
        SAFE_CALL(cudaMemcpy(lsh->h_hash_tables[i].buckets, temp_buckets, num_buckets * sizeof(Bucket_GPU), cudaMemcpyHostToDevice));
        free(temp_buckets);
    }
    
    // Copy hash tables to GPU
    SAFE_CALL(cudaMemcpy(lsh->hash_tables, lsh->h_hash_tables, L * sizeof(HashTable_GPU), cudaMemcpyHostToDevice));
    
    // Allocate working arrays on GPU
    SAFE_CALL(cudaMalloc(&lsh->d_candidate_flags, num_points * sizeof(int)));
    SAFE_CALL(cudaMalloc(&lsh->d_candidates_buffer, num_points * sizeof(int)));
    SAFE_CALL(cudaMalloc(&lsh->d_candidate_count, sizeof(int)));
    
    // Generate hyperplanes
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((lsh->num_hyperplanes * dimensions + block.x - 1) / block.x);
    
    generate_hyperplanes_kernel<<<grid, block>>>(lsh->hyperplanes, lsh->num_hyperplanes, dimensions, SEED);
    SAFE_CALL(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR;
    
    lsh->gpu_initialized = true;
    return lsh;
}

// Build LSH index on GPU
extern "C" void build_index_gpu(LSH_GPU *lsh) {
    if (!lsh->gpu_initialized) return;
    
    // Allocate temporary hash values array
    int *d_hash_values;
    SAFE_CALL(cudaMalloc(&d_hash_values, lsh->num_points * lsh->L * sizeof(int)));
    
    // Compute hash values for all points
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((lsh->num_points + block.x - 1) / block.x);
    
    compute_hashes_kernel<<<grid, block>>>(lsh->points, lsh->hyperplanes, d_hash_values, 
                                           lsh->num_points, lsh->L, lsh->k, lsh->dimensions);
    SAFE_CALL(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR;
    
    // Copy hash values to host for bucket population
    int *h_hash_values = (int*)malloc(lsh->num_points * lsh->L * sizeof(int));
    SAFE_CALL(cudaMemcpy(h_hash_values, d_hash_values, lsh->num_points * lsh->L * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Populate buckets on host first, then copy to GPU
    for (int table_idx = 0; table_idx < lsh->L; table_idx++) {
        int num_buckets = lsh->h_hash_tables[table_idx].num_buckets;
        Bucket_GPU *temp_buckets = (Bucket_GPU*)malloc(num_buckets * sizeof(Bucket_GPU));
        
        // Initialize bucket counts
        for (int i = 0; i < num_buckets; i++) {
            temp_buckets[i].count = 0;
            temp_buckets[i].capacity = (lsh->num_points / num_buckets) * 1.5;
            if (temp_buckets[i].capacity < 10) temp_buckets[i].capacity = 10;
        }
        
        // Count points per bucket
        for (size_t point_idx = 0; point_idx < lsh->num_points; point_idx++) {
            int hash_key = h_hash_values[point_idx * lsh->L + table_idx] % num_buckets;
            temp_buckets[hash_key].count++;
        }
        
        // Allocate point indices arrays
        for (int i = 0; i < num_buckets; i++) {
            if (temp_buckets[i].count > temp_buckets[i].capacity) {
                temp_buckets[i].capacity = temp_buckets[i].count * 1.2;
            }
            temp_buckets[i].point_indices = (int*)malloc(temp_buckets[i].capacity * sizeof(int));
            temp_buckets[i].count = 0; // Reset for actual population
        }
        
        // Populate buckets
        for (size_t point_idx = 0; point_idx < lsh->num_points; point_idx++) {
            int hash_key = h_hash_values[point_idx * lsh->L + table_idx] % num_buckets;
            temp_buckets[hash_key].point_indices[temp_buckets[hash_key].count++] = point_idx;
        }
        
        // Copy bucket data to GPU
        for (int i = 0; i < num_buckets; i++) {
            Bucket_GPU gpu_bucket;
            SAFE_CALL(cudaMalloc(&gpu_bucket.point_indices, temp_buckets[i].capacity * sizeof(int)));
            SAFE_CALL(cudaMemcpy(gpu_bucket.point_indices, temp_buckets[i].point_indices, 
                               temp_buckets[i].count * sizeof(int), cudaMemcpyHostToDevice));
            gpu_bucket.count = temp_buckets[i].count;
            gpu_bucket.capacity = temp_buckets[i].capacity;
            
            SAFE_CALL(cudaMemcpy(&lsh->h_hash_tables[table_idx].buckets[i], &gpu_bucket, 
                               sizeof(Bucket_GPU), cudaMemcpyHostToDevice));
            
            free(temp_buckets[i].point_indices);
        }
        
        free(temp_buckets);
    }
    
    free(h_hash_values);
    SAFE_CALL(cudaFree(d_hash_values));
}

// Find nearest neighbors using LSH on GPU
extern "C" int lsh_nearest_neighbors_gpu(LSH_GPU *lsh, Point_GPU *query_point, double max_distance, int *neighbors, double *distances) {
    if (!lsh->gpu_initialized) return 0;
    
    // Copy query point to GPU
    Point_GPU *d_query;
    SAFE_CALL(cudaMalloc(&d_query, sizeof(Point_GPU)));
    
    double *d_query_coords;
    SAFE_CALL(cudaMalloc(&d_query_coords, lsh->dimensions * sizeof(double)));
    SAFE_CALL(cudaMemcpy(d_query_coords, query_point->coords, lsh->dimensions * sizeof(double), cudaMemcpyHostToDevice));
    
    Point_GPU temp_query = {d_query_coords, query_point->id};
    SAFE_CALL(cudaMemcpy(d_query, &temp_query, sizeof(Point_GPU), cudaMemcpyHostToDevice));
    
    // Clear candidate flags and count
    SAFE_CALL(cudaMemset(lsh->d_candidate_flags, 0, lsh->num_points * sizeof(int)));
    SAFE_CALL(cudaMemset(lsh->d_candidate_count, 0, sizeof(int)));
    
    // Find candidates
    dim3 block1(THREADS_PER_BLOCK);
    dim3 grid1((lsh->L + block1.x - 1) / block1.x);
    
    find_candidates_kernel<<<grid1, block1>>>(lsh->hyperplanes, lsh->hash_tables, d_query, 
                                              lsh->d_candidates_buffer, lsh->d_candidate_count, lsh->d_candidate_flags,
                                              lsh->L, lsh->k, lsh->dimensions);
    SAFE_CALL(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR;
    
    // Get candidate count
    int candidate_count;
    SAFE_CALL(cudaMemcpy(&candidate_count, lsh->d_candidate_count, sizeof(int), cudaMemcpyDeviceToHost));
    
    if (candidate_count == 0) {
        SAFE_CALL(cudaFree(d_query_coords));
        SAFE_CALL(cudaFree(d_query));
        return 0;
    }
    
    // Allocate GPU memory for results
    int *d_neighbors, *d_neighbor_count;
    double *d_distances;
    SAFE_CALL(cudaMalloc(&d_neighbors, candidate_count * sizeof(int)));
    SAFE_CALL(cudaMalloc(&d_distances, candidate_count * sizeof(double)));
    SAFE_CALL(cudaMalloc(&d_neighbor_count, sizeof(int)));
    SAFE_CALL(cudaMemset(d_neighbor_count, 0, sizeof(int)));
    
    // Calculate distances and filter neighbors
    dim3 block2(THREADS_PER_BLOCK);
    dim3 grid2((candidate_count + block2.x - 1) / block2.x);
    
    calculate_distances_kernel<<<grid2, block2>>>(lsh->points, d_query, lsh->d_candidates_buffer, candidate_count, 
                                                 max_distance, d_neighbors, d_distances, d_neighbor_count, lsh->dimensions);
    SAFE_CALL(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR;
    
    // Get final neighbor count
    int neighbor_count;
    SAFE_CALL(cudaMemcpy(&neighbor_count, d_neighbor_count, sizeof(int), cudaMemcpyDeviceToHost));
    
    // Copy results back to host
    if (neighbor_count > 0) {
        SAFE_CALL(cudaMemcpy(neighbors, d_neighbors, neighbor_count * sizeof(int), cudaMemcpyDeviceToHost));
        SAFE_CALL(cudaMemcpy(distances, d_distances, neighbor_count * sizeof(double), cudaMemcpyDeviceToHost));
    }
    
    // Cleanup
    SAFE_CALL(cudaFree(d_query_coords));
    SAFE_CALL(cudaFree(d_query));
    SAFE_CALL(cudaFree(d_neighbors));
    SAFE_CALL(cudaFree(d_distances));
    SAFE_CALL(cudaFree(d_neighbor_count));
    
    return neighbor_count;
}

// Host function to calculate distance (for compatibility)
extern "C" double calculate_distance_gpu(Point_GPU *p1, Point_GPU *p2, int dimensions) {
    double sum = 0.0;
    for (int d = 0; d < dimensions; d++) {
        double diff = p1->coords[d] - p2->coords[d];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// Clean up GPU memory
extern "C" void cleanup_lsh_gpu(LSH_GPU *lsh) {
    if (!lsh->gpu_initialized) return;
    
    // Free point coordinates on GPU
    for (size_t i = 0; i < lsh->num_points; i++) {
        Point_GPU temp_point;
        SAFE_CALL(cudaMemcpy(&temp_point, &lsh->points[i], sizeof(Point_GPU), cudaMemcpyDeviceToHost));
        SAFE_CALL(cudaFree(temp_point.coords));
    }
    
    // Free points array
    SAFE_CALL(cudaFree(lsh->points));
    
    // Free hyperplanes
    SAFE_CALL(cudaFree(lsh->hyperplanes));
    free(lsh->h_hyperplanes);
    
    // Free hash table buckets
    for (int i = 0; i < lsh->L; i++) {
        for (int j = 0; j < lsh->h_hash_tables[i].num_buckets; j++) {
            Bucket_GPU temp_bucket;
            SAFE_CALL(cudaMemcpy(&temp_bucket, &lsh->h_hash_tables[i].buckets[j], sizeof(Bucket_GPU), cudaMemcpyDeviceToHost));
            SAFE_CALL(cudaFree(temp_bucket.point_indices));
        }
        SAFE_CALL(cudaFree(lsh->h_hash_tables[i].buckets));
    }
    
    // Free hash tables
    SAFE_CALL(cudaFree(lsh->hash_tables));
    free(lsh->h_hash_tables);
    
    // Free working arrays
    SAFE_CALL(cudaFree(lsh->d_candidate_flags));
    SAFE_CALL(cudaFree(lsh->d_candidates_buffer));
    SAFE_CALL(cudaFree(lsh->d_candidate_count));
    
    free(lsh);
}