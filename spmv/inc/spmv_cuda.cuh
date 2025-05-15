#ifndef _SMDV_CUDA_HH
#define _SMDV_CUDA_HH

#include "utils.cuh"

struct SmdvInfo {
    float *d_dst;
    float *d_val;
    float *d_arr;
    int *d_row;
    int *d_col;
    struct DeviceInfo dim;
};


float mul_cuda (
    float *dst, int *row, int *col, float *val, float *arr,
    int num_rows, int num_cols, int val_num
);

#endif