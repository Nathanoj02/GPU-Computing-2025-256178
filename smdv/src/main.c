#include "data.h"
#include "smdv.h"
#include "smdv_cuda.cuh"

#include <malloc.h>

int main()
{
    unsigned int row[] = {0, 0, 1, 3, 4, 5, 5};
    unsigned int col[] = {1, 2, 3, 0, 1, 0, 2};
    float val[] = {1, 2, 5, 1, 3, 4, 2};

    float arr[] = {2, 6, 1, 3};

    float *res = (float *) malloc(sizeof(float) * 6);

    mul(res, row, col, val, arr, 6, 7);

    // for (int i = 0; i < 6; i++) {
    //     printf("%.0lf ", res[i]);
    // }

    printf("\n\n");

    return 0;
}