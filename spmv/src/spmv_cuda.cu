#include "spmv_cuda.cuh"
#include "error.cuh"
#include "utils.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <malloc.h>
#include <math.h>

#include <sys/time.h>

SmdvInfo init_mul (size_t num_rows, size_t num_cols, size_t val_num);
float exec_mul (
    float *dst, size_t *row, size_t *col, float *val, float *arr,
    size_t num_rows, size_t num_cols, size_t val_num, SmdvInfo& smdv_info
);
void deinit_mul (SmdvInfo &smdv_info);



__global__
void mul_kernel (
    float *dst, size_t *row, size_t *col, float *val, float *arr, size_t val_num
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread is in range
    if (idx >= val_num)
        return;

    float value = val[idx] * arr[col[idx]];
    int row_idx = row[idx];
    
    atomicAdd(&dst[row_idx], value);
}


float mul_cuda (
    float *dst, size_t *row, size_t *col, float *val, float *arr,
    size_t num_rows, size_t num_cols, size_t val_num
)
{
    auto smdv_info = init_mul(num_rows, num_cols, val_num);
    float time_spent = exec_mul(dst, row, col, val, arr, num_rows, num_cols, val_num, smdv_info);
    deinit_mul(smdv_info);

    return time_spent;
}


float exec_mul (
    float *dst, size_t *row, size_t *col, float *val, float *arr,
    size_t num_rows, size_t num_cols, size_t val_num, SmdvInfo& smdv_info
) 
{
    // Memcpy
    SAFE_CALL( cudaMemcpy(smdv_info.d_val, val, sizeof(float) * val_num, cudaMemcpyHostToDevice) );
    SAFE_CALL( cudaMemcpy(smdv_info.d_arr, arr, sizeof(float) * num_cols, cudaMemcpyHostToDevice) );
    SAFE_CALL( cudaMemcpy(smdv_info.d_row, row, sizeof(size_t) * val_num, cudaMemcpyHostToDevice) );
    SAFE_CALL( cudaMemcpy(smdv_info.d_col, col, sizeof(size_t) * val_num, cudaMemcpyHostToDevice) );

    dim3 dim_grid = dim3(smdv_info.dim.grid.x, smdv_info.dim.grid.y, smdv_info.dim.grid.z);
    dim3 dim_block = dim3(smdv_info.dim.block.x, smdv_info.dim.block.y, smdv_info.dim.block.z);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Kernel invocation
    mul_kernel <<< dim_grid, dim_block >>> (
        smdv_info.d_dst, smdv_info.d_row, smdv_info.d_col, smdv_info.d_val, smdv_info.d_arr, val_num
    );

    CHECK_CUDA_ERROR

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Memcpy to Host
    SAFE_CALL( cudaMemcpy(dst, smdv_info.d_dst, sizeof(float) * num_rows, cudaMemcpyDeviceToHost) );

    return milliseconds * 1000; // Return in nanoseconds to match cpu algorithms
}


SmdvInfo init_mul (size_t num_rows, size_t num_cols, size_t val_num)
{
    SmdvInfo smdv_info;

    // Malloc
    SAFE_CALL( cudaMalloc(&smdv_info.d_dst, sizeof(float) * num_rows) );
    SAFE_CALL( cudaMemset(smdv_info.d_dst, 0, sizeof(float) * num_rows) );
    SAFE_CALL( cudaMalloc(&smdv_info.d_val, sizeof(float) * val_num) );
    SAFE_CALL( cudaMalloc(&smdv_info.d_arr, sizeof(float) * num_cols) );
    SAFE_CALL( cudaMalloc(&smdv_info.d_row, sizeof(size_t) * val_num) );
    SAFE_CALL( cudaMalloc(&smdv_info.d_col, sizeof(size_t) * val_num) );

    find_best_grid_linear(smdv_info.dim, val_num);

    return smdv_info;
}


void deinit_mul (SmdvInfo &smdv_info)
{
    // Free
    SAFE_CALL( cudaFree(smdv_info.d_dst) );
    SAFE_CALL( cudaFree(smdv_info.d_val) );
    SAFE_CALL( cudaFree(smdv_info.d_arr) );
    SAFE_CALL( cudaFree(smdv_info.d_row) );
    SAFE_CALL( cudaFree(smdv_info.d_col) );
}


