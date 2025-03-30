#include "spmv_cuda.cuh"
#include "error.cuh"
#include "utils.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <malloc.h>
#include <math.h>

#include <sys/time.h>

SmdvInfoOrdered init_mul (size_t num_rows, size_t num_cols, size_t val_num);
void exec_mul (
    float *dst, size_t *col, float *val, float *arr, size_t *thread_start,
    size_t num_rows, size_t num_cols, size_t val_num, SmdvInfoOrdered& smdv_info
);
void deinit_mul (SmdvInfoOrdered &smdv_info);


__global__
void mul_kernel (
    float *dst, size_t *col, float *val, float *arr, size_t *thread_start,
    size_t num_rows, size_t val_num
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread is in range
    if (idx >= num_rows)
        return;

    float p = 0;

    int thread_end = thread_start[idx + 1];
    for (int i = thread_start[idx]; i < thread_end; i++)
    {
        p += val[i] * arr[col[i]];
    }

    dst[idx] = p;
}


void mul_cuda_ordered (
    float *dst, size_t *col, float *val, float *arr, size_t *thread_start,
    size_t num_rows, size_t num_cols, size_t val_num
)
{
    // struct timeval t1 = {0, 0}, t2 = {0, 0};
    // gettimeofday(&t1, (struct timezone *) 0);

    auto smdv_info = init_mul(num_rows, num_cols, val_num);

    // gettimeofday(&t2, (struct timezone *) 0);

    // double time1 = ((t2.tv_sec - t1.tv_sec) * 1.e6 + (t2.tv_usec - t1.tv_usec));

    // printf("\nInit: %lf usec\n", time1);
    
    // gettimeofday(&t1, (struct timezone *) 0);
    
    
    exec_mul(dst, col, val, arr, thread_start, num_rows, num_cols, val_num, smdv_info);
    
    // gettimeofday(&t2, (struct timezone *) 0);
    
    // double time2 = ((t2.tv_sec - t1.tv_sec) * 1.e6 + (t2.tv_usec - t1.tv_usec));
    // printf("Exec: %lf usec\n", time2);
    
    // gettimeofday(&t1, (struct timezone *) 0);
    deinit_mul(smdv_info);
    // gettimeofday(&t2, (struct timezone *) 0);
    // double time3 = ((t2.tv_sec - t1.tv_sec) * 1.e6 + (t2.tv_usec - t1.tv_usec));
    // printf("End: %lf usec\n", time3);
}


void exec_mul (
    float *dst, size_t *col, float *val, float *arr, size_t *thread_start,
    size_t num_rows, size_t num_cols, size_t val_num, SmdvInfoOrdered& smdv_info
)
{
    // Memcpy
    SAFE_CALL( cudaMemcpy(smdv_info.d_val, val, sizeof(float) * val_num, cudaMemcpyHostToDevice) );
    SAFE_CALL( cudaMemcpy(smdv_info.d_arr, arr, sizeof(float) * num_cols, cudaMemcpyHostToDevice) );
    SAFE_CALL( cudaMemcpy(smdv_info.d_col, col, sizeof(size_t) * val_num, cudaMemcpyHostToDevice) );
    SAFE_CALL( cudaMemcpy(smdv_info.d_thread_start, thread_start, sizeof(size_t) * (num_rows + 1), cudaMemcpyHostToDevice) );

    dim3 dim_grid = dim3(smdv_info.dim.grid.x, smdv_info.dim.grid.y, smdv_info.dim.grid.z);
    dim3 dim_block = dim3(smdv_info.dim.block.x, smdv_info.dim.block.y, smdv_info.dim.block.z);

    // Kernel invocation
    mul_kernel <<< dim_grid, dim_block >>> (
        smdv_info.d_dst, smdv_info.d_col, smdv_info.d_val, smdv_info.d_arr, 
        smdv_info.d_thread_start, num_rows, val_num
    );

    // Memcpy to Host
    SAFE_CALL( cudaMemcpy(dst, smdv_info.d_dst, sizeof(float) * num_rows, cudaMemcpyDeviceToHost) );
}


SmdvInfoOrdered init_mul (size_t num_rows, size_t num_cols, size_t val_num)
{
    SmdvInfoOrdered smdv_info;

    // Malloc
    SAFE_CALL( cudaMalloc(&smdv_info.d_dst, sizeof(float) * num_rows) );
    SAFE_CALL( cudaMalloc(&smdv_info.d_val, sizeof(float) * val_num) );
    SAFE_CALL( cudaMalloc(&smdv_info.d_arr, sizeof(float) * num_cols) );
    SAFE_CALL( cudaMalloc(&smdv_info.d_col, sizeof(size_t) * val_num) );
    SAFE_CALL( cudaMalloc(&smdv_info.d_thread_start, sizeof(size_t) * (num_rows + 1)) );

    find_best_grid_linear(smdv_info.dim, num_rows);

    return smdv_info;
}


void deinit_mul (SmdvInfoOrdered &smdv_info)
{
    // Free
    SAFE_CALL( cudaFree(smdv_info.d_dst) );
    SAFE_CALL( cudaFree(smdv_info.d_val) );
    SAFE_CALL( cudaFree(smdv_info.d_arr) );
    SAFE_CALL( cudaFree(smdv_info.d_col) );
    SAFE_CALL( cudaFree(smdv_info.d_thread_start) );
}


