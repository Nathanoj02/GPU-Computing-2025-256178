#ifndef _DATA_HH
#define _DATA_HH

#include <stddef.h>

typedef struct DataOrdered
{
    size_t *row;
    size_t *col;
    float *val;
    float *arr;
    float *res;
    size_t *thread_start;
    size_t row_num;
    size_t col_num;
    size_t val_num;
} DataOrdered;


DataOrdered test_data_ordered();

int read_from_file_ordered(char *path, DataOrdered &data);

bool check_data(float *check, float *base, size_t size);

void mergeSort(size_t *row, size_t *col, float *val, int left, int right);

#endif