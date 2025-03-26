#ifndef _DATA_HH
#define _DATA_HH

#include <stddef.h>

typedef struct Data {
    size_t *row;
    size_t *col;
    float *val;
    float *arr;
    float *res;
    size_t *thread_start;
    size_t row_num;
    size_t col_num;
    size_t val_num;
} Data;

Data test_data();

Data read_from_file(char *path);

#endif