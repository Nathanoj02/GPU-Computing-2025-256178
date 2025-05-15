#include "data_cuda.cuh"

#include "spmv_cuda.cuh"
#include "spmv_cpu.cuh"

#include <stdio.h>
#include <sys/time.h>
#include <string>
#include <iostream>

#define WARM_UPS 3
#define RUNS 10

void printTimes(double times[]);

int main(int argc, char *argv[])
{
    DataCuda data;

    std::string filename;

    // Parse command line arguments
    if (argc > 1) {
        filename = argv[1];
    }

    // Load data
    if (filename.empty()) {
        data = test_data();
    }
    else {
        int success = read_from_file(&filename[0], &data);
        if (success < 0) 
            return -1;
    }
    
    // Just for checking values in the end
    float *cpu_res = (float *) calloc(data.row_num, sizeof(float));
    mul(cpu_res, data.row, data.col, data.val, data.arr, data.row_num, data.val_num);

    double times[RUNS];

    for (int i = -WARM_UPS; i < RUNS; i++)
    {
        times[i] = mul_cuda(data.res, data.row, data.col, data.val, data.arr, data.row_num, data.col_num, data.val_num);
    }
    
    printTimes(times);

    bool checkData = check_data(data.res, cpu_res, data.row_num);
    printf("Data is %scorrect\n\n", checkData ? "" : "NOT ");

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