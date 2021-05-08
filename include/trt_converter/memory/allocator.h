#ifndef INCLUDE_TRT_CONVERTER_MEMORY_ALLOCATOR_
#define INCLUDE_TRT_CONVERTER_MEMORY_ALLOCATOR_

#include "trt_converter/common/macro.h"

#include <cstddef>
#include <cstdint>

class Allocator {
  virtual void* Alloc(size_t size) = 0;
  virtual void Free(void* ptr) = 0;
};

class CpuAllocator : public Allocator {
  void* Alloc(size_t size) override {
    char* data = new char[size];
    return reinterpret_cast<void*>(data);
  }
  void Free(void* ptr) override { free(ptr); }
  template <typename T, typename... Args>
  T* Alloc(Args&&... args) {
    T* place = reinterpret_cast<T*>(Alloc(sizeof(T)));
    return new (place) T(std::forward<Args>(args)...);
  }
  template <typename T>
  void Free(T* ptr) {
    ptr.~T();
    Free((void*)ptr);
  }
};

class GpuAllocator : public Allocator {
  void* Alloc(size_t size) override {
    char* data;
    CUDA_CHECK(cudaMalloc((void**)&data, size));
    return data;
  }
  void Free(void* ptr) override { CUDA_CHECK(cudaFree(ptr)); }
};

#endif /* INCLUDE_TRT_CONVERTER_MEMORY_ALLOCATOR_ */
