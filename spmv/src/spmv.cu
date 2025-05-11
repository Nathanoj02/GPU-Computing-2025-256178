#include "spmv.cuh"
#include "data.cuh"

#include <stdio.h>

void mul(
    float *dst, size_t *row, size_t *col, float *val, float *arr,
    size_t num_rows, size_t val_num
)
{
    // Set all elements to 0
    for (int i = 0; i < num_rows; i++)
    {
        dst[i] = 0;
    }
    
    // Loop through row array
    for (int i = 0; i < val_num; i++)
    {   
        dst[row[i]] += val[i] * arr[col[i]];
    }
}


// First improvement
void mul_sorted(
    float *dst, size_t *row, size_t *col, float *val, float *arr,
    size_t num_rows, size_t val_num
)
{
    // Sort by row
    mergeSort(row, col, val, 0, val_num - 1);

    // Set all elements to 0
    for (int i = 0; i < num_rows; i++)
    {
        dst[i] = 0;
    }
    
    // Loop through row array
    for (int i = 0; i < val_num; i++)
    {   
        dst[row[i]] += val[i] * arr[col[i]];
    }
}

// Second improvement -> tiled MM
void mul_tiled(
    float *dst, size_t *row, size_t *col, float *val, float *arr,
    size_t num_rows, size_t val_num
)
{
    mergeSort(row, col, val, 0, val_num - 1);

    // Set all elements to 0
    for (int i = 0; i < num_rows; i++)
    {
        dst[i] = 0;
    }

    size_t TILE = 128*1024;
    for (size_t t = 0; t < val_num; t += TILE) {
        size_t t_end = std::min(val_num, t + TILE);

        for (size_t i = t; i < t_end; i++)
            dst[row[i]] += val[i] * arr[col[i]];
    }
}