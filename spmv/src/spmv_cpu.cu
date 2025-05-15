#include "spmv_cpu.cuh"
#include "data_cuda.cuh"

void mul(
    float *dst, int *row, int *col, float *val, float *arr,
    int num_rows, int val_num
)
{
    // Loop through row array
    for (int i = 0; i < val_num; i++)
    {   
        dst[row[i]] += val[i] * arr[col[i]];
    }
}