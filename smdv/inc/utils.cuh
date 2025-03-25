#ifndef _UTILS_CUDA_HH
#define _UTILS_CUDA_HH

struct DimInfo {
    size_t x;
    size_t y;
    size_t z;
};

struct DeviceInfo {
    DimInfo block;
    DimInfo grid;
};

DeviceInfo& find_best_grid_linear(DeviceInfo& device_info, size_t size);

#endif