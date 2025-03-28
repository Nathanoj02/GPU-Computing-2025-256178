#ifndef _SMDV_CUDA_HH
#define _SMDV_CUDA_HH

#include "utils.cuh"

struct SmdvInfoOrdered {
    float *d_dst;
    float *d_val;
    float *d_arr;
    size_t *d_col;
    size_t *d_thread_start;
    struct DeviceInfo dim;
};

void mul_cuda_ordered (
    float *dst, size_t *col, float *val, float *arr, size_t *thread_start,
    size_t num_rows, size_t num_cols, size_t val_num
);

#endif