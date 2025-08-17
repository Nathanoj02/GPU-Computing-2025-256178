#ifndef __NN_SEARCH_H__
#define __NN_SEARCH_H__

int naive_nearest_neighbors(
    LSH *lsh, Point *query_point, double max_distance, 
    int *neighbors, double *distances
);

#endif // __NN_SEARCH_H__
