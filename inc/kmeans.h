#ifndef KMEANS_H
#define KMEANS_H

#include <stdint.h>
#include <stddef.h>

typedef struct {
    uint8_t* dst;
    uint8_t* img;
    size_t img_height;
    size_t img_width;
    unsigned int k;
    unsigned int dimensions;
    float stab_error;
    int max_iterations;
} KMeansParams;

#endif // KMEANS_H