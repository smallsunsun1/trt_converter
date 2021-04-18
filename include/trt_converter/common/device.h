#ifndef INCLUDE_TRT_CONVERTER_COMMON_DEVICE_
#define INCLUDE_TRT_CONVERTER_COMMON_DEVICE_

#include <chrono>
#include <iostream>
#include <thread>

#include "cuda_runtime_api.h"

namespace sss {

inline void CudaStatusCheck(cudaError_t status) {
    if (status != cudaSuccess) {
        std::cerr << "cuda error: " << cudaGetErrorString(status) << std::endl;
    }
}

#if CUDA_VERSION < 10000
inline void CudaSleep(cudaStream_t stream, cudaError_t status, void* sleep)
#else
inline void CudaSleep(void* sleep)
#endif
{
  std::this_thread::sleep_for(std::chrono::duration<int, std::milli>(*static_cast<int*>(sleep)));
}

class TRTCudaEvent;

class TRTCudaStream {
 public:
  TRTCudaStream() { CudaStatusCheck(cudaStreamCreate(&stream_)); }
  TRTCudaStream(const TRTCudaStream&) = delete;
  TRTCudaStream& operator=(const TRTCudaStream&) = delete;
  TRTCudaStream(TRTCudaStream&&) = delete;
  TRTCudaStream& operator=(TRTCudaStream&&) = delete;
  ~TRTCudaStream() {CudaStatusCheck(cudaStreamDestroy(stream_)); }
  void Synchronize() { CudaStatusCheck(cudaStreamSynchronize(stream_)); }
  void Wait(TRTCudaEvent& event);
  cudaStream_t Get() const { return stream_; }
  void Sleep(int* ms) {
    CudaStatusCheck(cudaStreamAddCallback(stream_, CudaSleep, ms, 0));
  }

 private:
  cudaStream_t stream_{};
};

class TRTCudaEvent {
public:
    TRTCudaEvent() {CudaStatusCheck(cudaEventCreate(&event_));}
    TRTCudaEvent(const TRTCudaEvent&) = delete;
    TRTCudaEvent& operator=(const TRTCudaEvent&) = delete;
    TRTCudaEvent(TRTCudaEvent&&) = delete;
    TRTCudaEvent& operator=(TRTCudaEvent&&) = delete;
    ~TRTCudaEvent() {CudaStatusCheck(cudaEventDestroy(event_));}
    cudaEvent_t Get() const {return event_;}
    void Record(cudaStream_t stream) {CudaStatusCheck(cudaEventRecord(event_, stream));}
    void Synchronize() {CudaStatusCheck(cudaEventSynchronize(event_));}
    float operator-(const TRTCudaEvent& other) {
        float time{0};
        CudaStatusCheck(cudaEventElapsedTime(&time, other.Get(), Get()));
        return time;
    }
private:
    cudaEvent_t event_{};
};

inline void TRTCudaStream::Wait(TRTCudaEvent &event) {
    CudaStatusCheck(cudaStreamWaitEvent(stream_, event.Get(), 0));
}

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
    void Launch(TRTCudaStream& stream) {
        CudaStatusCheck(cudaGraphLaunch(executor_, stream.Get()));
    }
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

}  // namespace sss

#endif /* INCLUDE_TRT_CONVERTER_COMMON_DEVICE_ */
