#include "smdv.h"

#include <stdio.h>

void mul(
    float *dst, unsigned int *row, unsigned int *col, float *val, float *arr,
    unsigned int num_rows, unsigned int val_num
)
{
    // Set all elements to 0
    for (int i = 0; i < num_rows; i++)
    {
        dst[i] = 0;
    }

    float p = 0;    // tracks product value
    // Loop through row array
    for (int i = 0; i < val_num; i++)
    {
        // If row is different from last one save result of multiplication
        if (i > 0 && row[i] != row[i - 1]) 
        {
            dst[row[i - 1]] = p;
            p = 0;
        }
        
        p += val[i] * arr[col[i]];
    }

    // Write last value
    dst[num_rows - 1] = p;
}