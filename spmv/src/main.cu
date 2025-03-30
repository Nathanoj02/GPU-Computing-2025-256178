#include "data.cuh"
#include "spmv.cuh"

#include "spmv_cuda.cuh"

#include <stdio.h>
#include <sys/time.h>

int main(int argc, char *argv[])
{
    DataOrdered data;

    if (argc == 1) {
        data = test_data_ordered();
    } 
    else {
        int success = read_from_file_ordered(argv[1], data);
        
        if (success < 0) 
        return -1;
    }

    float *cpu_res = (float *) malloc(sizeof(float) * data.row_num);
    mul(cpu_res, data.row, data.col, data.val, data.arr, data.row_num, data.val_num);
    
    // for (int i = 0; i < data.val_num; i++) {
    //     if (i < 10 || i > data.val_num - 10)
    //         printf("(%ld, %ld) -> %f\n", data.row[i], data.col[i], data.val[i]);
    // }

    mergeSort(data.row, data.col, data.val, 0, data.val_num - 1);

    // printf("\n\n");
    // for (int i = 0; i < data.val_num; i++) {
    //     printf("(%ld, %ld) -> %f\n", data.row[i], data.col[i], data.val[i]);
    // }

    // printf("\n--------------------------------------------------------------------\nThread start:\n");
    // for (int i = 0; i < data.row_num + 1; i++) {
    //     printf("%ld\n", data.thread_start[i]);
    // }
        
    struct timeval t1 = {0, 0}, t2 = {0, 0};

    double time = 0;

    for (int i = 0; i < 13; i++)
    {
        gettimeofday(&t1, (struct timezone *) 0);
        
        
        mul_cuda_ordered(data.res, data.col, data.val, data.arr, data.thread_start, data.row_num, data.col_num, data.val_num);
        // mul(data.res, data.row, data.col, data.val, data.arr, data.row_num, data.val_num);

    
        gettimeofday(&t2, (struct timezone *) 0);
    
        if (i > 2) {
            time += ((t2.tv_sec - t1.tv_sec) * 1.e6 + (t2.tv_usec - t1.tv_usec));
        }
    }
    
    printf("\nElapsed time: %lf usec\n", time / 10);
    printf("\t\t%lf s\n", (time / 10) / 1.e6);
    
    // for (int i = 0; i < data.row_num; i++) {
    //     printf("%.2lf\n", data.res[i]);
    // }


    bool checkData = check_data(data.res, cpu_res, data.row_num);
    printf("Data is %scorrect\n\n", checkData ? "" : "NOT ");

    return 0;
}