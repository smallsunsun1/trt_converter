#ifndef INCLUDE_TRT_CONVERTER_COMMON_UTILS_
#define INCLUDE_TRT_CONVERTER_COMMON_UTILS_

#include <cuda.h>

#include <algorithm>
#include <fstream>
#include <numeric>
#include <random>
#include <unordered_map>

#include "NvInfer.h"
#include "device.h"
#if CUDA_VERSION < 10000
#include <half.h>
#else
#include <cuda_fp16.h>
#endif

namespace sss {

inline int DataTypeSize(nvinfer1::DataType data_type) {
  switch (data_type) {
    case nvinfer1::DataType::kINT32:
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kINT8:
      return 1;
  }
  return 0;
}

template <typename T>
inline T RoundUp(T m, T n) {
  return ((m + n - 1) / n) * n;
}

template <typename A, typename B>
inline A DivUp(A x, B n) {
  return (x + n - 1) / n;
}

inline int Volume(const nvinfer1::Dims& d) { return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int>()); }

inline int Volume(const nvinfer1::Dims& dims, const nvinfer1::Dims& strides, int vec_dim, int comps, int batch) {
  int maxNbElems = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    // Get effective length of axis.
    int d = dims.d[i];
    // Any dimension is 0, it is an empty tensor.
    if (d == 0) {
      return 0;
    }
    if (i == vec_dim) {
      d = DivUp(d, comps);
    }
    maxNbElems = std::max(maxNbElems, d * strides.d[i]);
  }
  return maxNbElems * batch * (vec_dim < 0 ? 1 : comps);
}

inline int Volume(nvinfer1::Dims dims, int vec_dim, int comps, int batch) {
  if (vec_dim != -1) {
    dims.d[vec_dim] = RoundUp(dims.d[vec_dim], comps);
  }
  return Volume(dims) * std::max(batch, 1);
}

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

template <typename T>
inline void DumpBuffer(const void* buffer, uint32_t volume, const std::string& seperator, std::ostream& os) {
  const T* type_buffer = reinterpret_cast<const T*>(buffer);
  std::string sep;
  for (uint32_t v = 0; v < volume; ++v) {
    os << sep << type_buffer[v];
    sep = seperator;
  }
}

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
  void Fill() {
    switch (data_type) {
      case nvinfer1::DataType::kBOOL: {
        FillBuffer(buffer.GetHostBuffer(), volume, 0, 1);
        break;
      }
      case nvinfer1::DataType::kINT8: {
        FillBuffer(buffer.GetHostBuffer(), volume, std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
        break;
      }
      case nvinfer1::DataType::kINT32: {
        FillBuffer(buffer.GetHostBuffer(), volume, -128, 127);
        break;
      }
      case nvinfer1::DataType::kHALF: {
#if CUDA_VERSION < 10000
        FillBuffer<half_float::half>(buffer.GetHostBuffer(), volume, static_cast<half_float::half>(-1.0), static_cast<half_float::half>(-1.0));
#else
        FillBuffer<__half>(buffer.GetHostBuffer(), volume, -1.0, 1.0);
#endif
        break;
      }
      case nvinfer1::DataType::kFLOAT: {
        FillBuffer(buffer.GetHostBuffer(), volume, -1.0f, 1.0f);
        break;
      }
    }
  }
  void Dump(std::ostream& os, const std::string separator = " ") {
    switch (data_type) {
      case nvinfer1::DataType::kBOOL: {
        DumpBuffer<bool>(buffer.GetHostBuffer(), volume, separator, os);
        break;
      }
      case nvinfer1::DataType::kINT32: {
        DumpBuffer<int32_t>(buffer.GetHostBuffer(), volume, separator, os);
        break;
      }
      case nvinfer1::DataType::kINT8: {
        DumpBuffer<int8_t>(buffer.GetHostBuffer(), volume, separator, os);
        break;
      }
      case nvinfer1::DataType::kFLOAT: {
        DumpBuffer<float>(buffer.GetHostBuffer(), volume, separator, os);
        break;
      }
      case nvinfer1::DataType::kHALF: {
#if CUDA_VERSION < 10000
        DumpBuffer<half_float::half>(buffer.getHostBuffer(), volume, separator, os);
#else
        DumpBuffer<__half>(buffer.GetHostBuffer(), volume, separator, os);
#endif
        break;
      }
    }
  }
};

class Bindings {
 public:
  void AddBinding(int b, const std::string& name, bool is_input, int volume, nvinfer1::DataType data_type, const std::string& filename = "") {
    bindings_.resize(b);
    device_pointers_.resize(b);
    names_[name] = b;
    bindings_[b].is_input = is_input;
    bindings_[b].buffer.Allocate(static_cast<size_t>(volume) * static_cast<size_t>(DataTypeSize(data_type)));
  }

 private:
  std::unordered_map<std::string, int> names_;
  std::vector<Binding> bindings_;
  std::vector<void*> device_pointers_;
};

template <typename T>
struct TRTObjDeleter {
  void operator()(T* obj) { obj->destroy(); }
};

template <typename T>
using TRTUniquePtr = std::unique_ptr<T, TRTObjDeleter<T>>;

};  // namespace sss

#endif /* INCLUDE_TRT_CONVERTER_COMMON_UTILS_ */
