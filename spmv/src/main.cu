#include "data.cuh"
#include "spmv.cuh"

#include "spmv_cuda.cuh"

#include <stdio.h>
#include <sys/time.h>
#include <string>

int main(int argc, char *argv[])
{
    Data data;

    bool profile = false;
    std::string filename;


    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--profile" || arg == "-p") {
            profile = true;
        }
        else {
            filename = arg;
        }
    }

    // Load data
    if (filename.empty()) {
        data = test_data();
    }
    else {
        int success = read_from_file(&filename[0], data);
        if (success < 0) 
            return -1;
    }
    
    // Check if want to profile or not
    if (profile) {
        float *cpu_res = (float *) malloc(sizeof(float) * data.row_num);
        mul(cpu_res, data.row, data.col, data.val, data.arr, data.row_num, data.val_num);
    }
    else {
        // Just for checking values in the end
        float *cpu_res = (float *) malloc(sizeof(float) * data.row_num);
        mul(cpu_res, data.row, data.col, data.val, data.arr, data.row_num, data.val_num);

        struct timeval t1 = {0, 0}, t2 = {0, 0};

        double time = 0;

        for (int i = 0; i < 13; i++)
        {
            gettimeofday(&t1, (struct timezone *) 0);
            
            mul_cuda(data.res, data.row, data.col, data.val, data.arr, data.row_num, data.col_num, data.val_num);
            // mul_cuda_ordered(data.res, data.col, data.val, data.arr, data.thread_start, data.row_num, data.col_num, data.val_num);
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
    }

    return 0;
}