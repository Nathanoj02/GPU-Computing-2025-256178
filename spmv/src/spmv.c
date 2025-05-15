#include "spmv.h"
#include "data.h"

#include <stdio.h>
#include <stdlib.h>

void mul(
    float *dst, int *row, int *col, float *val, float *arr,
    int num_rows, int val_num
)
{
    for (int i = 0; i < val_num; i++)
    {   
        dst[row[i]] += val[i] * arr[col[i]];
    }
}