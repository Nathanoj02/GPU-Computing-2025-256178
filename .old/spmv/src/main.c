#include "data.h"

#include "spmv.h"

#include <stdio.h>
#include <sys/time.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>


#define WARM_UPS 3
#define RUNS 10

void printTimes(double times[]);

int main(int argc, char *argv[])
{
    Data data;

    bool profile = false;
    int alg = -1;
    char *filename = NULL;
    bool found_file = false;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        char *arg = argv[i];
        
        if (strcmp(arg, "--profile") == 0 || strcmp(arg, "-p") == 0) {
            profile = true;
        }
        else if (strcmp(arg, "--alg") == 0) {
            if (alg != -1) {
                printf("Error: --alg cannot be combined with other CPU algorithm options.\n");
                return 1;
            }
            alg = 0;
        }
        else if (strcmp(arg, "--alg1") == 0) {
            if (alg != -1) {
                printf("Error: --alg1 cannot be combined with other CPU algorithm options.\n");
                return 1;
            }
            alg = 1;
        }
        else if (strcmp(arg, "--alg2") == 0) {
            if (alg != -1) {
                printf("Error: --alg2 cannot be combined with other CPU algorithm options.\n");
                return 1;
            }
            alg = 2;
        }
        else if (!found_file) {
            filename = arg;
            found_file = true;
        }
    }

    if (alg == -1)
        alg = 0;

    // --------------- End line arguments ---------------------

    // Load data
    if (!found_file) {
        data = test_data();
    }
    else {
        int success = read_from_file(filename, &data);
        if (success < 0) 
            return -1;
    }
    
    // Check if want to profile or not
    if (profile) {
        if (alg >= 1) {
            mergeSort(data.row, data.col, data.val, 0, data.val_num - 1);
        }

        switch (alg)
        {
            case 0 :
                mul(data.res, data.row, data.col, data.val, data.arr, data.val_num);
                break;
            case 1 :
                mul(data.res, data.row, data.col, data.val, data.arr, data.val_num);
                break;
            default:
                break;
        }
    }
    else {
        struct timeval t1 = {0, 0}, t2 = {0, 0};

        double times[RUNS];
        
        if (alg == 1) {
            mergeSort(data.row, data.col, data.val, 0, data.val_num - 1);
        }

        for (int i = -WARM_UPS; i < RUNS; i++)
        {
            // Reset res array for CPU algorithms
            for (int j = 0; j < data.row_num; j++) {
                data.res[j] = 0;
            }

            switch (alg) {
                case 0 :    // cpu naive
                case 1:     // cpu sorted
                    gettimeofday(&t1, (struct timezone *) 0);
                    mul(data.res, data.row, data.col, data.val, data.arr, data.val_num);
                    gettimeofday(&t2, (struct timezone *) 0);
                    break;
                case 2: // openMP
                    gettimeofday(&t1, (struct timezone *) 0);
                    mul_omp(data.res, data.row, data.col, data.val, data.arr, data.val_num, data.row_num);
                    gettimeofday(&t2, (struct timezone *) 0);
                default :
                    return -1;
            }
        
            if (i >= 0) {
                times[i] = ((t2.tv_sec - t1.tv_sec) * 1.e6 + (t2.tv_usec - t1.tv_usec));
            }
        }
        
        printTimes(times);
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
    
    // printf("\nArithmetic mean time: %lf usec\n", ar_mean);
    // printf("\t\t%lf s\n", ar_mean / 1.e6);

    printf("\nStd: %lf usec\n", std_time);
}