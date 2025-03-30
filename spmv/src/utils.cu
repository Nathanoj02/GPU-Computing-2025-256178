#include "utils.cuh"
#include "error.cuh"

static cudaDeviceProp find_best_gpu()
{
    // Save properties of GPUs
    int dev_count;
    SAFE_CALL( cudaGetDeviceCount(&dev_count) );

    // Save best GPU
    cudaDeviceProp dev_prop;
    cudaDeviceProp best_device_prop;

    for (int i = 0; i < dev_count; i++)
    {
        SAFE_CALL( cudaGetDeviceProperties(&dev_prop, i) );
        
        if (i == 0 
            || (dev_prop.maxThreadsPerMultiProcessor * dev_prop.multiProcessorCount > 
                best_device_prop.maxThreadsPerMultiProcessor * best_device_prop.multiProcessorCount))
        {
            best_device_prop = dev_prop;
        }
    }
    return best_device_prop;
}


DeviceInfo& find_best_grid_linear(DeviceInfo& device_info, size_t size)
{
    auto best_device = find_best_gpu();

    std::size_t threadsPerBlock;

    if(size > best_device.maxThreadsPerBlock)
        threadsPerBlock = best_device.maxThreadsPerBlock;
    else
        threadsPerBlock = size;

    device_info.block = {
        threadsPerBlock,
        1,
        1
    };
    device_info.grid = {
        size_t(ceil(size / (float) threadsPerBlock)),
        1,
        1
    };
    
    return device_info;
}