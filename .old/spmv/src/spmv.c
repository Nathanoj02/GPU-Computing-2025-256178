#include "spmv.h"
#include "data.h"

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void mul(
    float *dst, int *row, int *col, float *val, float *arr, int val_num
)
{
    for (int i = 0; i < val_num; i++)
    {   
        dst[row[i]] += val[i] * arr[col[i]];
    }
}

void mul_omp(
    float *dst, int *row, int *col, float *val, float *arr, int val_num, int row_num
)
{
    #pragma omp parallel
    {
        float *dst_private = (float *) calloc(row_num, sizeof(float));
        
        #pragma omp for nowait
        for (int i = 0; i < val_num; i++) {
            dst_private[row[i]] += val[i] * arr[col[i]];
        }
        
        #pragma omp critical
        {
            for (int i = 0; i < row_num; i++) {
                dst[i] += dst_private[i];
            }
        }

        free(dst_private);
    }
}