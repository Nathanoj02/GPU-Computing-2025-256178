#include "data.cuh"
#include "spmv.cuh"

#include "spmv_cuda.cuh"

#include <stdio.h>
#include <sys/time.h>
#include <string>
#include <iostream>

#define WARM_UPS 3
#define RUNS 10

void printTimes(double times[]);

int main(int argc, char *argv[])
{
    Data data;

    bool profile = false;
    int alg = 0;
    std::string filename;


    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--profile" || arg == "-p") {
            profile = true;
        }
        else if (arg == "--cpu") {
            if (alg != 0) {
                std::cerr << "Error: --cpu cannot be combined with other CPU algorithm options.\n";
                return 1;
            }
            alg = 1;
        }
        else if (arg == "--cpu1") {
            if (alg != 0) {
                std::cerr << "Error: --cpu1 cannot be combined with other CPU algorithm options.\n";
                return 1;
            }
            alg = 2;
        }
        else if (arg == "--cpu2") {
            if (alg != 0) {
                std::cerr << "Error: --cpu2 cannot be combined with other CPU algorithm options.\n";
                return 1;
            }
            alg = 3;
        }
        else {
            filename = arg;
        }
    }
    // --------------- End line arguments ---------------------

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
        if (alg >= 2) {
            mergeSort(data.row, data.col, data.val, 0, data.val_num - 1);
        }

        switch (alg)
        {
            case 0 :
            case 1 :
                mul(data.res, data.row, data.col, data.val, data.arr, data.row_num, data.val_num);
                break;
            case 2 :
                mul_sorted(data.res, data.row, data.col, data.val, data.arr, data.row_num, data.val_num);
                break;
            case 3 :
                mul_tiled(data.res, data.row, data.col, data.val, data.arr, data.row_num, data.val_num);
                break;
            default:
                break;
        }
    }
    else {
        // Just for checking values in the end
        float *cpu_res = (float *) malloc(sizeof(float) * data.row_num);
        mul(cpu_res, data.row, data.col, data.val, data.arr, data.row_num, data.val_num);

        struct timeval t1 = {0, 0}, t2 = {0, 0};

        double times[RUNS];
        double cuda_time;

        for (int i = -WARM_UPS; i < RUNS; i++)
        {
            if (alg >= 2) {
                mergeSort(data.row, data.col, data.val, 0, data.val_num - 1);
            }

            switch (alg) {
                case 0 :    // gpu
                    cuda_time = mul_cuda(data.res, data.row, data.col, data.val, data.arr, data.row_num, data.col_num, data.val_num);
                    break;
                case 1 :    // cpu naive
                    gettimeofday(&t1, (struct timezone *) 0);
                    mul(data.res, data.row, data.col, data.val, data.arr, data.row_num, data.val_num);
                    gettimeofday(&t2, (struct timezone *) 0);
                    break;
                case 2 :    // cpu sorted
                    gettimeofday(&t1, (struct timezone *) 0);
                    mul_sorted(data.res, data.row, data.col, data.val, data.arr, data.row_num, data.val_num);
                    gettimeofday(&t2, (struct timezone *) 0);
                    break;
                case 3 :    // cpu tiled
                    gettimeofday(&t1, (struct timezone *) 0);
                    mul_tiled(data.res, data.row, data.col, data.val, data.arr, data.row_num, data.val_num);
                    gettimeofday(&t2, (struct timezone *) 0);
                    break;
                default :
                    break;
            }
        
            if (i >= 0) {
                if (alg > 0)
                    times[i] = ((t2.tv_sec - t1.tv_sec) * 1.e6 + (t2.tv_usec - t1.tv_usec));
                else
                    times[i] = cuda_time;
            }
        }
        
        printTimes(times);

        bool checkData = check_data(data.res, cpu_res, data.row_num);
        printf("Data is %scorrect\n\n", checkData ? "" : "NOT ");
    }

    return 0;
}

void printTimes(double times[]) {
    double geo_mean = 1;
    double ar_mean = 0;
    double std_time = 0;

    for (int i = 0; i < RUNS; i++) {
        geo_mean *= times[i];
        ar_mean += times[i];
    }
    geo_mean = pow(geo_mean, 1.0 / RUNS);
    ar_mean /= RUNS;

    for (int i = 0; i < RUNS; i++) {
        std_time += pow(times[i] - ar_mean, 2);
    }

    std_time = sqrt(std_time / RUNS);

    printf("\nGeometric mean time: %lf usec\n", geo_mean);
    printf("\t\t%lf s\n", geo_mean / 1.e6);
    
    printf("\nArithmetic mean time: %lf usec\n", ar_mean);
    printf("\t\t%lf s\n", ar_mean / 1.e6);

    printf("\nStd: %lf usec\n", std_time);
}