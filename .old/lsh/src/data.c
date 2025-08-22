#include "data.h"

Point* read_data(size_t *num_points, const char *filename) {
    // Check if file exists
    FILE *file = fopen(filename, "r");
    if (!file) {
        return NULL;
    }

    // Read first line to get first the number of dimensions and then number of points
    size_t dimensions; 

    if (fscanf(file, "%zu %zu", &dimensions, num_points) != 2) {
        fclose(file);
        return NULL;
    }

    // Read points from the file
    Point *points = malloc(*num_points * sizeof(Point));
    
    for (size_t i = 0; i < *num_points; i++) {
        points[i].coords = malloc(dimensions * sizeof(double));
        
        for (size_t j = 0; j < dimensions; j++) {
            if (fscanf(file, "%lf", &points[i].coords[j]) != 1) {
                fclose(file);
                return NULL;
            }
        }

        // Assign an ID based on index
        points[i].id = i;
    }

    fclose(file);

    return points;
} 