#include "lsh.h"

// Initialize a bucket
void init_bucket(Bucket *bucket, int expected_capacity) {
    bucket->point_indices = (int*)malloc(expected_capacity * sizeof(int));
    bucket->count = 0;
    bucket->capacity = expected_capacity;
}

// Initialize hash table with 2^k buckets
void init_hash_table(HashTable *table, int k, size_t num_points) {
    int num_buckets = 1 << k;
    table->num_buckets = num_buckets;
    table->buckets = malloc(num_buckets * sizeof(Bucket));
    
    int expected_capacity = num_points / num_buckets + 10;
    for (int i = 0; i < num_buckets; i++) {
        init_bucket(&table->buckets[i], expected_capacity);
    }
}

// Add point index to specific bucket
static inline void add_to_bucket(HashTable *table, int hash_value, int point_index) {
    Bucket *bucket = &table->buckets[hash_value];
    
    // Resize if needed (rare)
    if (bucket->count >= bucket->capacity) {
        bucket->capacity *= 2;
        bucket->point_indices = (int*)realloc(bucket->point_indices, 
                                            bucket->capacity * sizeof(int));
    }
    
    // Add point index
    bucket->point_indices[bucket->count] = point_index;
    bucket->count++;
}

// Get points from specific bucket
static inline int* get_from_bucket(HashTable *table, int hash_value, int *count) {
    Bucket *bucket = &table->buckets[hash_value];
    *count = bucket->count;
    return bucket->point_indices;
}

// Generate L*k hyperplanes
void generate_hyperplanes(LSH *lsh) {
    srand(SEED);   // Fixed seed for reproducible results

    int total_hp = lsh->L * lsh->k;

    for (int i = 0; i < total_hp; i++) {
        double norm = 0.0;
        
        // Generate random normal vector
        for (int d = 0; d < lsh->dimensions; d++) {
            lsh->hyperplanes[i][d] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;  // Random [-1, 1]
            norm += lsh->hyperplanes[i][d] * lsh->hyperplanes[i][d];
        }
        
        // Normalize the hyperplane
        norm = sqrt(norm);
        for (int d = 0; d < lsh->dimensions; d++) {
            lsh->hyperplanes[i][d] /= norm;
        }
    }
}

// Compute multi-bit hash for a table
int compute_table_hash(LSH *lsh, Point *point, int table_idx) {
    int hash = 0;
    int base = table_idx * lsh->k;
    for (int h = 0; h < lsh->k; h++) {
        double dot = 0.0;
        for (int d = 0; d < lsh->dimensions; d++) {
            dot += point->coords[d] * lsh->hyperplanes[base+h][d];
        }
        hash = (hash << 1) | (dot >= 0);
    }
    return hash;
}

// Initialize LSH structure
LSH* init_lsh(int dimensions, size_t L, size_t k, Point *points, size_t num_points) {
    LSH *lsh = (LSH *) malloc(sizeof(LSH));
    lsh->L = L;  // Use L tables for given hyperplanes
    lsh->k = k;  // Use k hyperplanes per table

    lsh->points = points;
    lsh->num_points = num_points;
    lsh->dimensions = dimensions;
    lsh->num_hyperplanes = L * k;

    // Allocate hash tables
    lsh->hash_tables = (HashTable *) malloc(L * sizeof(HashTable));

    // Allocate hyperplanes
    lsh->hyperplanes = (double **) malloc(lsh->num_hyperplanes * sizeof(double *));
    for (int i = 0; i < lsh->num_hyperplanes; i++) {
        lsh->hyperplanes[i] = (double *) malloc(dimensions * sizeof(double));
    }

    // Pre-allocate reusable arrays
    lsh->temp_hashes = (int *) malloc(lsh->num_hyperplanes * sizeof(int));
    lsh->candidate_flags = NULL;  // Will be allocated based on num_points

    lsh->candidates_buffer = (int *) malloc(num_points * sizeof(int));
    
    // Initialize hash tables
    for (int i = 0; i < lsh->L; i++) {
        init_hash_table(&lsh->hash_tables[i], lsh->k, num_points);
    }
    
    // Generate hyperplanes
    generate_hyperplanes(lsh);
    
    return lsh;
}

// Build LSH index with multi-bit hashing
void build_index(LSH *lsh) {
    // Allocate candidate flags array once
    lsh->candidate_flags = (int *)calloc(lsh->num_points, sizeof(int));
    
    // Hash each point and add to appropriate buckets
    for (size_t i = 0; i < lsh->num_points; i++) {
        for (int j = 0; j < lsh->L; j++) {
            int hash_key = compute_table_hash(lsh, &lsh->points[i], j);
            Bucket *bucket = &lsh->hash_tables[j].buckets[hash_key];
            
            // Resize bucket if needed
            if (bucket->count >= bucket->capacity) {
                bucket->capacity *= 2;
                bucket->point_indices = (int *)realloc(bucket->point_indices, 
                                                     bucket->capacity * sizeof(int));
            }
            
            // Add point index to bucket
            bucket->point_indices[bucket->count] = i;
            bucket->count++;
        }
    }
}

// Find candidate points for a query
static int find_candidates(LSH *lsh, Point *query, int *candidates) {
    // Clear candidate flags from previous query
    static int prev_candidate_count = 0;
    for (int i = 0; i < prev_candidate_count; i++) {
        lsh->candidate_flags[candidates[i]] = 0;
    }
    
    int candidate_count = 0;
    
    // Check each hash table
    for (int j = 0; j < lsh->L; j++) {
        int hash_key = compute_table_hash(lsh, query, j);
        Bucket *bucket = &lsh->hash_tables[j].buckets[hash_key];
        
        // Process all points in this bucket
        for (int i = 0; i < bucket->count; i++) {
            int point_idx = bucket->point_indices[i];
            
            // Mark as candidate if not already
            if (!lsh->candidate_flags[point_idx]) {
                lsh->candidate_flags[point_idx] = 1;
                candidates[candidate_count] = point_idx;
                candidate_count++;
            }
        }
    }
    
    prev_candidate_count = candidate_count;
    return candidate_count;
}

// Calculate distance between two points
double calculate_distance(Point *p1, Point *p2, int dimensions) {
    double sum = 0.0;
    for (int d = 0; d < dimensions; d++) {
        double diff = p1->coords[d] - p2->coords[d];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// Find nearest neighbors using LSH
int lsh_nearest_neighbors(
    LSH *lsh, Point *query_point,
    double max_distance, int *neighbors, double *distances
) {
    int *candidates = lsh->candidates_buffer;
    int candidate_count = find_candidates(lsh, query_point, candidates);
    
    int neighbor_count = 0;
    
    // Check distances for candidates only
    for (int i = 0; i < candidate_count; i++) {
        int point_idx = candidates[i];
        Point *point = &lsh->points[point_idx];
        double dist = calculate_distance(query_point, point, lsh->dimensions);
        
        // Ensure we don't include the query point itself
        if (point->id != query_point->id && dist <= max_distance) {
            neighbors[neighbor_count] = point_idx;
            distances[neighbor_count] = dist;
            neighbor_count++;
        }
    }
    
    return neighbor_count;
}

// Clean up memory
void cleanup_lsh(LSH *lsh) {
    // Clean up buckets
    for (int h = 0; h < lsh->L; h++) {
        for (int b = 0; b < lsh->hash_tables[h].num_buckets; b++) {
            free(lsh->hash_tables[h].buckets[b].point_indices);
        }
    free(lsh->hash_tables[h].buckets);
}
    
    // Free hyperplanes
    for (int i = 0; i < lsh->num_hyperplanes; i++) {
        free(lsh->hyperplanes[i]);
    }
    free(lsh->hyperplanes);
    
    // Free hash tables
    free(lsh->hash_tables);
    
    // Free pre-allocated arrays
    free(lsh->temp_hashes);
    free(lsh->candidate_flags);
    free(lsh->candidates_buffer);

    free(lsh);
}