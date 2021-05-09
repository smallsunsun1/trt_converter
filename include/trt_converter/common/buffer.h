#ifndef INCLUDE_TRT_CONVERTER_COMMON_BUFFER_
#define INCLUDE_TRT_CONVERTER_COMMON_BUFFER_

#include <cassert>
#include <iostream>
#include <iterator>
#include <memory>
#include <new>
#include <numeric>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "trt_converter/common/common.h"
#include "trt_converter/common/half.h"
#include "trt_converter/common/utils.h"

namespace sss {

//!
//! \brief  The GenericBuffer class is a templated class for buffers.
//!
//! \details This templated RAII (Resource Acquisition Is Initialization) class handles the allocation,
//!          deallocation, querying of buffers on both the device and the host.
//!          It can handle data of arbitrary types because it stores byte buffers.
//!          The template parameters AllocFunc and FreeFunc are used for the
//!          allocation and deallocation of the buffer.
//!          AllocFunc must be a functor that takes in (void** ptr, size_t size)
//!          and returns bool. ptr is a pointer to where the allocated buffer address should be stored.
//!          size is the amount of memory in bytes to allocate.
//!          The boolean indicates whether or not the memory allocation was successful.
//!          FreeFunc must be a functor that takes in (void* ptr) and returns void.
//!          ptr is the allocated buffer address. It must work with nullptr input.
//!
template <typename AllocFunc, typename FreeFunc>
class GenericBuffer {
 public:
  //!
  //! \brief Construct an empty buffer.
  //!
  GenericBuffer(nvinfer1::DataType type = nvinfer1::DataType::kFLOAT)
      : size_(0), capacity_(0), type_(type), buffer_(nullptr) {}

  //!
  //! \brief Construct a buffer with the specified allocation size in bytes.
  //!
  GenericBuffer(size_t size, nvinfer1::DataType type) : size_(size), capacity_(size), type_(type) {
    if (!alloc_fn(&buffer_, this->nbBytes())) {
      throw std::bad_alloc();
    }
  }

  GenericBuffer(GenericBuffer&& buf)
      : size_(buf.size_), capacity_(buf.capacity_), type_(buf.type_), buffer_(buf.buffer_) {
    buf.size_ = 0;
    buf.capacity_ = 0;
    buf.type_ = nvinfer1::DataType::kFLOAT;
    buf.buffer_ = nullptr;
  }

  GenericBuffer& operator=(GenericBuffer&& buf) {
    if (this != &buf) {
      free_fn(buffer_);
      size_ = buf.size_;
      capacity_ = buf.capacity_;
      type_ = buf.type_;
      buffer_ = buf.buffer_;
      // Reset buf.
      buf.size_ = 0;
      buf.capacity_ = 0;
      buf.buffer_ = nullptr;
    }
    return *this;
  }

  //!
  //! \brief Returns pointer to underlying array.
  //!
  void* data() { return buffer_; }

  //!
  //! \brief Returns pointer to underlying array.
  //!
  const void* data() const { return buffer_; }

  //!
  //! \brief Returns the size (in number of elements) of the buffer.
  //!
  size_t size() const { return size_; }

  //!
  //! \brief Returns the size (in bytes) of the buffer.
  //!
  size_t nbBytes() const { return this->size() * GetElementSize(type_); }

  //!
  //! \brief Resizes the buffer. This is a no-op if the new size is smaller than or equal to the current capacity.
  //!
  void resize(size_t newSize) {
    size_ = newSize;
    if (capacity_ < newSize) {
      free_fn(buffer_);
      if (!alloc_fn(&buffer_, this->nbBytes())) {
        throw std::bad_alloc{};
      }
      capacity_ = newSize;
    }
  }

  //!
  //! \brief Overload of resize that accepts Dims
  //!
  void resize(const nvinfer1::Dims& dims) { return this->resize(Volume(dims)); }

  ~GenericBuffer() { free_fn(buffer_); }

 private:
  size_t size_{0}, capacity_{0};
  nvinfer1::DataType type_;
  void* buffer_;
  AllocFunc alloc_fn;
  FreeFunc free_fn;
};

class DeviceAllocator {
 public:
  bool operator()(void** ptr, size_t size) const { return cudaMalloc(ptr, size) == cudaSuccess; }
};

class DeviceFree {
 public:
  void operator()(void* ptr) const { cudaFree(ptr); }
};

class HostAllocator {
 public:
  bool operator()(void** ptr, size_t size) const {
    *ptr = malloc(size);
    return *ptr != nullptr;
  }
};

class HostFree {
 public:
  void operator()(void* ptr) const { free(ptr); }
};

using DeviceBuffer = GenericBuffer<DeviceAllocator, DeviceFree>;
using HostBuffer = GenericBuffer<HostAllocator, HostFree>;

//!
//! \brief  The ManagedBuffer class groups together a pair of corresponding device and host buffers.
//!
class ManagedBuffer {
 public:
  DeviceBuffer device_buffer;
  HostBuffer host_buffer;
};

//!
//! \brief  The BufferManager class handles host and device buffer allocation and deallocation.
//!
//! \details This RAII class handles host and device buffer allocation and deallocation,
//!          memcpy between host and device buffers to aid with inference,
//!          and debugging dumps to validate inference. The BufferManager class is meant to be
//!          used to simplify buffer management and any interactions between buffers and the engine.
//!
class BufferManager {
 public:
  static constexpr size_t kInvalidSizeValue = ~size_t(0);

  //!
  //! \brief Create a BufferManager for handling buffer interactions with engine.
  //!
  BufferManager(std::shared_ptr<nvinfer1::ICudaEngine> engine, const int batch_size = 0,
                const nvinfer1::IExecutionContext* context = nullptr)
      : engine_(engine), batch_size_(batch_size) {
    // Full Dims implies no batch size.
    assert(engine->hasImplicitBatchDimension() || batch_size_ == 0);
    // Create host and device buffers
    for (int i = 0; i < engine_->getNbBindings(); i++) {
      auto dims = context ? context->getBindingDimensions(i) : engine_->getBindingDimensions(i);
      size_t vol = context || !batch_size_ ? 1 : static_cast<size_t>(batch_size_);
      nvinfer1::DataType type = engine_->getBindingDataType(i);
      int vecDim = engine_->getBindingVectorizedDim(i);
      if (-1 != vecDim)  // i.e., 0 != lgScalarsPerVector
      {
        int scalarsPerVec = engine_->getBindingComponentsPerElement(i);
        dims.d[vecDim] = DivUp(dims.d[vecDim], scalarsPerVec);
        vol *= scalarsPerVec;
      }
      vol *= Volume(dims);
      std::unique_ptr<ManagedBuffer> manBuf{new ManagedBuffer()};
      manBuf->device_buffer = DeviceBuffer(vol, type);
      manBuf->host_buffer = HostBuffer(vol, type);
      device_bindings_.emplace_back(manBuf->device_buffer.data());
      managed_buffers_.emplace_back(std::move(manBuf));
    }
  }

  //!
  //! \brief Returns a vector of device buffers that you can use directly as
  //!        bindings for the execute and enqueue methods of IExecutionContext.
  //!
  std::vector<void*>& getDeviceBindings() { return device_bindings_; }

  //!
  //! \brief Returns a vector of device buffers.
  //!
  const std::vector<void*>& getDeviceBindings() const { return device_bindings_; }

  //!
  //! \brief Returns the device buffer corresponding to tensorName.
  //!        Returns nullptr if no such tensor can be found.
  //!
  void* getDeviceBuffer(const std::string& tensorName) const { return getBuffer(false, tensorName); }

  //!
  //! \brief Returns the host buffer corresponding to tensorName.
  //!        Returns nullptr if no such tensor can be found.
  //!
  void* getHostBuffer(const std::string& tensorName) const { return getBuffer(true, tensorName); }

  //!
  //! \brief Returns the size of the host and device buffers that correspond to tensorName.
  //!        Returns kINVALID_SIZE_VALUE if no such tensor can be found.
  //!
  size_t size(const std::string& tensor_name) const {
    int index = engine_->getBindingIndex(tensor_name.c_str());
    if (index == -1) return kInvalidSizeValue;
    return managed_buffers_[index]->host_buffer.nbBytes();
  }

  //!
  //! \brief Dump host buffer with specified tensorName to ostream.
  //!        Prints error message to std::ostream if no such tensor can be found.
  //!
  void dumpBuffer(std::ostream& os, const std::string& tensorName) {
    int index = engine_->getBindingIndex(tensorName.c_str());
    if (index == -1) {
      os << "Invalid tensor name" << std::endl;
      return;
    }
    void* buf = managed_buffers_[index]->host_buffer.data();
    size_t bufSize = managed_buffers_[index]->host_buffer.nbBytes();
    nvinfer1::Dims bufDims = engine_->getBindingDimensions(index);
    size_t rowCount = static_cast<size_t>(bufDims.nbDims > 0 ? bufDims.d[bufDims.nbDims - 1] : batch_size_);
    int leadDim = batch_size_;
    int* trailDims = bufDims.d;
    int nbDims = bufDims.nbDims;

    // Fix explicit Dimension networks
    if (!leadDim && nbDims > 0) {
      leadDim = bufDims.d[0];
      ++trailDims;
      --nbDims;
    }

    os << "[" << leadDim;
    for (int i = 0; i < nbDims; i++) os << ", " << trailDims[i];
    os << "]" << std::endl;
    switch (engine_->getBindingDataType(index)) {
      case nvinfer1::DataType::kINT32:
        print<int32_t>(os, buf, bufSize, rowCount);
        break;
      case nvinfer1::DataType::kFLOAT:
        print<float>(os, buf, bufSize, rowCount);
        break;
      case nvinfer1::DataType::kHALF:
        print<half_float::half>(os, buf, bufSize, rowCount);
        break;
      case nvinfer1::DataType::kINT8:
        assert(0 && "Int8 network-level input and output is not supported");
        break;
      case nvinfer1::DataType::kBOOL:
        assert(0 && "Bool network-level input and output are not supported");
        break;
    }
  }

  //!
  //! \brief Templated print function that dumps buffers of arbitrary type to std::ostream.
  //!        rowCount parameter controls how many elements are on each line.
  //!        A rowCount of 1 means that there is only 1 element on each line.
  //!
  template <typename T>
  void print(std::ostream& os, void* buf, size_t bufSize, size_t rowCount) {
    assert(rowCount != 0);
    assert(bufSize % sizeof(T) == 0);
    T* typedBuf = static_cast<T*>(buf);
    size_t numItems = bufSize / sizeof(T);
    for (int i = 0; i < static_cast<int>(numItems); i++) {
      // Handle rowCount == 1 case
      if (rowCount == 1 && i != static_cast<int>(numItems) - 1)
        os << typedBuf[i] << std::endl;
      else if (rowCount == 1)
        os << typedBuf[i];
      // Handle rowCount > 1 case
      else if (i % rowCount == 0)
        os << typedBuf[i];
      else if (i % rowCount == rowCount - 1)
        os << " " << typedBuf[i] << std::endl;
      else
        os << " " << typedBuf[i];
    }
  }

  void copyInputToDevice() { memcpyBuffers(true, false, false); }

  void copyOutputToHost() { memcpyBuffers(false, true, false); }

  void copyInputToDeviceAsync(const cudaStream_t& stream = 0) { memcpyBuffers(true, false, true, stream); }

  void copyOutputToHostAsync(const cudaStream_t& stream = 0) { memcpyBuffers(false, true, true, stream); }

  ~BufferManager() = default;

 private:
  void* getBuffer(const bool isHost, const std::string& tensorName) const {
    int index = engine_->getBindingIndex(tensorName.c_str());
    if (index == -1) return nullptr;
    return (isHost ? managed_buffers_[index]->host_buffer.data() : managed_buffers_[index]->device_buffer.data());
  }

  void memcpyBuffers(const bool copy_input, const bool device_to_host, const bool async,
                     const cudaStream_t& stream = 0) {
    for (int i = 0; i < engine_->getNbBindings(); i++) {
      void* dst_ptr =
          device_to_host ? managed_buffers_[i]->host_buffer.data() : managed_buffers_[i]->device_buffer.data();
      const void* src_ptr =
          device_to_host ? managed_buffers_[i]->device_buffer.data() : managed_buffers_[i]->host_buffer.data();
      const size_t byte_size = managed_buffers_[i]->host_buffer.nbBytes();
      const cudaMemcpyKind memcpyType = device_to_host ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
      if ((copy_input && engine_->bindingIsInput(i)) || (!copy_input && !engine_->bindingIsInput(i))) {
        if (async)
          CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src_ptr, byte_size, memcpyType, stream));
        else
          CUDA_CHECK(cudaMemcpy(dst_ptr, src_ptr, byte_size, memcpyType));
      }
    }
  }

  std::shared_ptr<nvinfer1::ICudaEngine> engine_;                //!< The pointer to the engine
  int batch_size_;                                               //!< The batch size for legacy networks, 0 otherwise.
  std::vector<std::unique_ptr<ManagedBuffer>> managed_buffers_;  //!< The vector of pointers to managed buffers
  std::vector<void*> device_bindings_;  //!< The vector of device buffers needed for engine execution
};

}  // namespace sss

#endif /* INCLUDE_TRT_CONVERTER_COMMON_BUFFER_ */
