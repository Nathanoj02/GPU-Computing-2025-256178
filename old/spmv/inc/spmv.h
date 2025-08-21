#ifndef _SMDV_HH
#define _SMDV_HH

#include <stddef.h>

void mul(
    float *dst, int *row, int *col, float *val, float *arr, int val_num
);

void mul_omp(
    float *dst, int *row, int *col, float *val, float *arr, int val_num, int row_num
);

#endif