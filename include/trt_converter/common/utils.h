#ifndef INCLUDE_TRT_CONVERTER_COMMON_UTILS_
#define INCLUDE_TRT_CONVERTER_COMMON_UTILS_

#include <fstream>
#include <random>
#include <algorithm>
#include "NvInfer.h"
#include "device.h"
#include <cuda.h>
#if CUDA_VERSION < 10000
#include <half.h>
#else
#include <cuda_fp16.h>
#endif

namespace sss {

template <typename T>
inline void FillBuffer(void* buffer, int volume, T min, T max) {
    T* type_buffer = reinterpret_cast<T*>(buffer);
    std::default_random_engine engine;
    if constexpr (std::is_integral_v<T>) {
        std::uniform_int_distribution<int> dist(min, max);
        auto generator = [&engine, &dist]() { return static_cast<T>(dist(engine)); };
        std::generate(type_buffer, type_buffer + volume, generator);
    } else {
        std::uniform_real_distribution<float> dist(min, max);
        auto generator = [&engine, &dist]() { return static_cast<T>(dist(engine)); };
        std::generate(type_buffer, type_buffer + volume, generator);
    }
}
// template <>
// #if CUDA_VERSION < 10000
// inline void FillBuffer<half_float::half>(void* buffer, int volume, half_float::half min, half_float::half max)
// #else
// inline void FillBuffer<__half>(void* buffer, int volume, __half min, __half max)
// #endif
// {
//     fillBufferHalf(buffer, volume, min, max);
// }

struct Binding {
    bool is_input = false;
    MirroredBuffer buffer;
    uint32_t volume = 0;
    nvinfer1::DataType data_type = nvinfer1::DataType::kFLOAT;
    void Fill(const std::string& filename) {
        std::ifstream file(filename, std::ios::in | std::ios::binary);
        if (file.is_open()) {
            file.read(reinterpret_cast<char*>(buffer.GetHostBuffer()), buffer.GetSize());
            file.close();
        }
    }
};

};

#endif /* INCLUDE_TRT_CONVERTER_COMMON_UTILS_ */
