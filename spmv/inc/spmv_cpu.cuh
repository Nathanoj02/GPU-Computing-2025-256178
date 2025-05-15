#ifndef _SMDV_CPU_HH
#define _SMDV_CPU_HH

#include <stddef.h>

void mul(
    float *dst, int *row, int *col, float *val, float *arr,
    int num_rows, int val_num
);

#endif