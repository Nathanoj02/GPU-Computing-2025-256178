#ifndef _DATA_CUDA_HH
#define _DATA_CUDA_HH

#include <stddef.h>

typedef struct DataCuda
{
    int *row;
    int *col;
    float *val;
    float *arr;
    float *res;
    int row_num;
    int col_num;
    int val_num;
} DataCuda;


DataCuda test_data();

int read_from_file(char *path, DataCuda *data);

bool check_data(float *check, float *base, int size);

#endif