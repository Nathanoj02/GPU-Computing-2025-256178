#ifndef _DATA_HH
#define _DATA_HH

#include <stddef.h>

typedef struct Data
{
    size_t *row;
    size_t *col;
    float *val;
    float *arr;
    float *res;
    size_t row_num;
    size_t col_num;
    size_t val_num;
} Data;


Data test_data();

int read_from_file(char *path, Data &data);

bool check_data(float *check, float *base, size_t size);

void mergeSort(size_t *row, size_t *col, float *val, int left, int right);

#endif