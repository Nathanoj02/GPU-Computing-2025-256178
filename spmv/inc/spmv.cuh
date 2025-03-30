#ifndef _SMDV_HH
#define _SMDV_HH

#include <stddef.h>

void mul(
    float *dst, size_t *row, size_t *col, float *val, float *arr,
    size_t num_rows, size_t val_num
);

#endif