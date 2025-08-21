#ifndef _DATA_HH
#define _DATA_HH

#include <stddef.h>
#include <stdbool.h>

typedef struct Data
{
    int *row;
    int *col;
    float *val;
    float *arr;
    float *res;
    int row_num;
    int col_num;
    int val_num;
} Data;


Data test_data();

int read_from_file(char *path, Data *data);

bool check_data(float *check, float *base, int size);

void mergeSort(int *row, int *col, float *val, int left, int right);

#endif