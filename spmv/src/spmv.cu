#include "spmv.cuh"

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