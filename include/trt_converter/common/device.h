#ifndef INCLUDE_TRT_CONVERTER_COMMON_DEVICE_
#define INCLUDE_TRT_CONVERTER_COMMON_DEVICE_

#include <chrono>
#include <iostream>
#include <thread>

#include "cuda_runtime_api.h"
#include "trt_converter/memory/allocator.h"

namespace sss {

inline void CudaStatusCheck(cudaError_t status) {
  if (status != cudaSuccess) {
    std::cerr << "cuda error: " << cudaGetErrorString(status) << std::endl;
  }
}

#if CUDA_VERSION < 10000
inline void CudaSleep(cudaStream_t stream, cudaError_t status, void* sleep) {
  (void)stream;
  (void)status;
  std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(*static_cast<int*>(sleep)));
}
#else
inline void CudaSleep(void* sleep) { std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(*static_cast<int*>(sleep))); }
#endif

class TRTCudaEvent;

class TRTCudaStream {
 public:
  TRTCudaStream() { CudaStatusCheck(cudaStreamCreate(&stream_)); }
  TRTCudaStream(const TRTCudaStream&) = delete;
  TRTCudaStream& operator=(const TRTCudaStream&) = delete;
  TRTCudaStream(TRTCudaStream&&) = delete;
  TRTCudaStream& operator=(TRTCudaStream&&) = delete;
  ~TRTCudaStream() { CudaStatusCheck(cudaStreamDestroy(stream_)); }
  void Synchronize() { CudaStatusCheck(cudaStreamSynchronize(stream_)); }
  void Wait(TRTCudaEvent& event);
  cudaStream_t Get() const { return stream_; }
  void Sleep(int* ms) { CudaStatusCheck(cudaStreamAddCallback(stream_, CudaSleep, ms, 0)); }

 private:
  cudaStream_t stream_{};
};

class TRTCudaEvent {
 public:
  TRTCudaEvent() { CudaStatusCheck(cudaEventCreate(&event_)); }
  TRTCudaEvent(const TRTCudaEvent&) = delete;
  TRTCudaEvent& operator=(const TRTCudaEvent&) = delete;
  TRTCudaEvent(TRTCudaEvent&&) = delete;
  TRTCudaEvent& operator=(TRTCudaEvent&&) = delete;
  ~TRTCudaEvent() { CudaStatusCheck(cudaEventDestroy(event_)); }
  cudaEvent_t Get() const { return event_; }
  void Record(const TRTCudaStream& stream) { CudaStatusCheck(cudaEventRecord(event_, stream.Get())); }
  void Synchronize() { CudaStatusCheck(cudaEventSynchronize(event_)); }
  float operator-(const TRTCudaEvent& other) {
    float time{0};
    CudaStatusCheck(cudaEventElapsedTime(&time, other.Get(), Get()));
    return time;
  }

 private:
  cudaEvent_t event_{};
};

inline void TRTCudaStream::Wait(TRTCudaEvent& event) { CudaStatusCheck(cudaStreamWaitEvent(stream_, event.Get(), 0)); }

class TRTCudaGraph {
 public:
  TRTCudaGraph() = default;
  TRTCudaGraph(const TRTCudaGraph&) = delete;
  TRTCudaGraph& operator=(const TRTCudaGraph&) = delete;
  TRTCudaGraph(TRTCudaGraph&&) = delete;
  TRTCudaGraph& operator=(TRTCudaGraph&&) = delete;
  void BeginCapture(TRTCudaStream& stream) {
    CudaStatusCheck(cudaGraphCreate(&graph_, 0));
    CudaStatusCheck(cudaStreamBeginCapture(stream.Get(), cudaStreamCaptureModeThreadLocal));
  }
  void Launch(TRTCudaStream& stream) { CudaStatusCheck(cudaGraphLaunch(executor_, stream.Get())); }
  void EndCapture(TRTCudaStream& stream) {
    CudaStatusCheck(cudaStreamEndCapture(stream.Get(), &graph_));
    CudaStatusCheck(cudaGraphInstantiate(&executor_, graph_, nullptr, nullptr, 0));
    CudaStatusCheck(cudaGraphDestroy(graph_));
  }
  ~TRTCudaGraph() {
    if (executor_) {
      cudaGraphExecDestroy(executor_);
    }
  }

 private:
  cudaGraph_t graph_{};
  cudaGraphExec_t executor_{};
};

struct TRTHostAllocator {
  void operator()(void** ptr, size_t size) { CudaStatusCheck(cudaMallocHost(ptr, size)); }
};
struct TRTHostDeallocator {
  void operator()(void* ptr) { CudaStatusCheck(cudaFreeHost(ptr)); }
};
struct TRTDeviceAllocator {
  void operator()(void** ptr, size_t size) { CudaStatusCheck(cudaMalloc(ptr, size)); }
};
struct TRTDeviceDeallocator {
  void operator()(void* ptr) { CudaStatusCheck(cudaFree(ptr)); }
};

template <typename Allocator, typename DeAllocator>
class TRTCudaBuffer {
 public:
  TRTCudaBuffer() = default;
  TRTCudaBuffer(const TRTCudaBuffer&) = delete;
  TRTCudaBuffer& operator=(const TRTCudaBuffer) = delete;
  TRTCudaBuffer(TRTCudaBuffer&& other) {
    ptr_ = other.ptr_;
    other.ptr_ = nullptr;
  }
  TRTCudaBuffer& operator=(TRTCudaBuffer&& other) {
    if (this != &other) {
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }
  ~TRTCudaBuffer() { Reset(); }
  TRTCudaBuffer(size_t size) { Allocator()(&ptr_, size); }
  void Reset(void* ptr = nullptr) {
    if (ptr_) {
      DeAllocator()(ptr_);
    }
    ptr_ = ptr;
  }
  void Allocate(size_t size) {
    Reset();
    Allocator()(&ptr_, size);
  }
  void* Get() const { return ptr_; }

 private:
  void* ptr_ = nullptr;
};

using TRTDeviceBuffer = TRTCudaBuffer<TRTDeviceAllocator, TRTDeviceDeallocator>;
using TRTHostBuffer = TRTCudaBuffer<TRTHostAllocator, TRTHostDeallocator>;

class MirroredBuffer {
 public:
  MirroredBuffer() = default;
  MirroredBuffer(size_t size) : size_(size), device_buffer_(size), host_buffer_(size) {}
  size_t GetSize() const { return size_; }
  void Allocate(size_t size) {
    size_ = size;
    device_buffer_.Allocate(size);
    device_buffer_.Allocate(size);
  }
  void DeviceToHost(TRTCudaStream& stream) {
    CudaStatusCheck(cudaMemcpyAsync(host_buffer_.Get(), device_buffer_.Get(), size_, cudaMemcpyDeviceToHost, stream.Get()));
  }
  void HostToDevice(TRTCudaStream& stream) {
    CudaStatusCheck(cudaMemcpyAsync(device_buffer_.Get(), host_buffer_.Get(), size_, cudaMemcpyHostToDevice, stream.Get()));
  }
  void* GetDeviceBuffer() const { return device_buffer_.Get(); }
  void* GetHostBuffer() const { return host_buffer_.Get(); }

 private:
  size_t size_ = 0;
  TRTDeviceBuffer device_buffer_;
  TRTHostBuffer host_buffer_;
};

inline void setCudaDevice(int device, std::ostream& os) {
  CudaStatusCheck(cudaSetDevice(device));

  cudaDeviceProp properties;
  CudaStatusCheck(cudaGetDeviceProperties(&properties, device));

  // clang-format off
    os << "=== Device Information ===" << std::endl;
    os << "Selected Device: "      << properties.name                                               << std::endl;
    os << "Compute Capability: "   << properties.major << "." << properties.minor                   << std::endl;
    os << "SMs: "                  << properties.multiProcessorCount                                << std::endl;
    os << "Compute Clock Rate: "   << properties.clockRate / 1000000.0F << " GHz"                   << std::endl;
    os << "Device Global Memory: " << (properties.totalGlobalMem >> 20) << " MiB"                   << std::endl;
    os << "Shared Memory per SM: " << (properties.sharedMemPerMultiprocessor >> 10) << " KiB"       << std::endl;
    os << "Memory Bus Width: "     << properties.memoryBusWidth << " bits"
                        << " (ECC " << (properties.ECCEnabled != 0 ? "enabled" : "disabled") << ")" << std::endl;
    os << "Memory Clock Rate: "    << properties.memoryClockRate / 1000000.0F << " GHz"             << std::endl;
  // clang-format on
}

}  // namespace sss

#endif /* INCLUDE_TRT_CONVERTER_COMMON_DEVICE_ */
