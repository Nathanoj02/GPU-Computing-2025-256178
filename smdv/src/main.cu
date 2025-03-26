#include "data.cuh"
#include "smdv.cuh"

#include "smdv_cuda.cuh"

#include <stdio.h>
#include <sys/time.h>

int main(int argc, char *argv[])
{
    Data data;

    if (argc == 1) {
        data = test_data();
    } 
    else {
        data = read_from_file(argv[1]);
    }

    // for (int i = 0; i < data.val_num; i++) {
    //     printf("(%ld, %ld) -> %f\n", data.row[i], data.col[i], data.val[i]);
    // }

    // printf("\n--------------------------------------------------------------------\nThread start:\n");
    // for (int i = 0; i < data.row_num; i++) {
    //     printf("%ld\n", data.thread_start[i]);
    // }
        
    struct timeval t1 = {0, 0}, t2 = {0, 0};
    gettimeofday(&t1, (struct timezone *) 0);
    // mul(data.res, data.row, data.col, data.val, data.arr, data.row_num, data.val_num);
    
    mul_cuda(data.res, data.col, data.val, data.arr, data.thread_start, data.row_num, data.col_num, data.val_num);

    gettimeofday(&t2, (struct timezone *) 0);

    double time = ((t2.tv_sec - t1.tv_sec) * 1.e6 + (t2.tv_usec - t1.tv_usec));

    
    for (int i = 0; i < data.row_num; i++) {
        printf("%.2lf\n", data.res[i]);
    }
    
    printf("\nElapsed time: %lf\n", time);
    printf("\n\n");

    return 0;
}