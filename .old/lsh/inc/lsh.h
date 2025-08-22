#ifndef __LSH_H__
#define __LSH_H__

#include <stdlib.h>
#include <math.h>

#define SEED 0

#ifdef __cplusplus
extern "C" {
#endif

// Structure for a point in n-dimensional space
typedef struct {
    double *coords;
    int id;
} Point;

// Bucket structure
typedef struct {
    int *point_indices;     // Array of point indices in this bucket
    int count;              // Number of points in this bucket
    int capacity;           // Capacity of point_indices array
} Bucket;

// Hash table for one hyperplane
typedef struct {
    Bucket *buckets;
    int num_buckets;
} HashTable;

// LSH structure
typedef struct {
    Point *points;          // Array of all points
    size_t num_points;      // Number of points
    int dimensions;         // Dimensionality of points
    size_t num_hyperplanes; // Number of hyperplanes
    double **hyperplanes;   // Hyperplane normal vectors
    HashTable *hash_tables; // One hash table per hyperplane

    // Pre-allocated arrays to avoid malloc in hot paths
    int *temp_hashes;       // Reusable hash array
    int *candidate_flags;   // Reusable candidate tracking
    int *candidates_buffer;

    int L;             // Number of tables
    int k;             // Hyperplanes per table
} LSH;

LSH* init_lsh(int dimensions, size_t L, size_t k, Point *points, size_t num_points);
void build_index(LSH *lsh);
int lsh_nearest_neighbors(LSH *lsh, Point *query_point, double max_distance, int *neighbors, double *distances);
void cleanup_lsh(LSH *lsh);

double calculate_distance(Point *p1, Point *p2, int dimensions);

#ifdef __cplusplus
}
#endif

#endif // __LSH_H__