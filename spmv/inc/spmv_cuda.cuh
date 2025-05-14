#ifndef _SMDV_CUDA_HH
#define _SMDV_CUDA_HH

#include "utils.cuh"

struct SmdvInfo {
    float *d_dst;
    float *d_val;
    float *d_arr;
    size_t *d_row;
    size_t *d_col;
    struct DeviceInfo dim;
};


float mul_cuda (
    float *dst, size_t *row, size_t *col, float *val, float *arr,
    size_t num_rows, size_t num_cols, size_t val_num
);

#endif