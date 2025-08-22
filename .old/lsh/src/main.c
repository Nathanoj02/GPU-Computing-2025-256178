#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "data.h"
#include "lsh.h"
#include "nn_search.h"

#define NAME_LEN 256

int main(int argc, char *argv[]) {
    // Get dimensions , L and k from command line arguments
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

    // Check if file exists
    char filename[NAME_LEN];
    snprintf(filename, sizeof(filename), "dataset/points_%zud.txt", dimensions);

    size_t num_points;
    Point *points = read_data(&num_points, filename);

    if (!points) {
        fprintf(stderr, "Error reading data from file: %s\n", filename);
        return 1;
    }

    // Initialize LSH
    LSH *lsh = init_lsh(dimensions, L, k, points, num_points);

    // Build index
    build_index(lsh);

    // Take first one as query point
    Point query_point;
    query_point.coords = points[0].coords;
    query_point.id = points[0].id;

    int *lsh_neighbors = (int *) malloc(num_points * sizeof(int));
    double *lsh_distances = (double *) malloc(num_points * sizeof(double));

    // Check time of algorithm
    clock_t start = clock();
    int lsh_count = lsh_nearest_neighbors(lsh, &query_point, 0.2, lsh_neighbors, lsh_distances);
    clock_t end = clock();

    double lsh_time = (double)(end - start) / CLOCKS_PER_SEC;

    // Naive nearest neighbors for comparison
    int *naive_neighbors = (int *) malloc(num_points * sizeof(int));
    double *naive_distances = (double *) malloc(num_points * sizeof(double));
    
    start = clock();
    int naive_count = naive_nearest_neighbors(lsh, &query_point, 0.2, naive_neighbors, naive_distances);
    end = clock();

    double naive_time = (double)(end - start) / CLOCKS_PER_SEC;

    // Print times in milliseconds
    printf("LSH Time: %f milliseconds\n", lsh_time * 1000);
    printf("Naive Time: %f milliseconds\n", naive_time * 1000);


    // Cleanup
    cleanup_lsh(lsh);

    for (size_t i = 0; i < num_points; i++) {
        free(points[i].coords);
    }
    free(points);
    
    return 0;
}