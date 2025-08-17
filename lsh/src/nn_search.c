#include "lsh.h"
#include "nn_search.h"

// Naive nearest neighbor search
int naive_nearest_neighbors(LSH *lsh, Point *query_point, double max_distance, 
                           int *neighbors, double *distances) {
    int neighbor_count = 0;
    
    for (int i = 0; i < lsh->num_points; i++) {
        double dist = calculate_distance(query_point, &lsh->points[i], lsh->dimensions);
        
        if (lsh->points[i].id != query_point->id && dist <= max_distance) {
            neighbors[neighbor_count] = i;
            distances[neighbor_count] = dist;
            neighbor_count++;
        }
    }
    
    return neighbor_count;
}