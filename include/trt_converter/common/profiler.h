#ifndef INCLUDE_TRT_CONVERTER_COMMON_PROFILER_
#define INCLUDE_TRT_CONVERTER_COMMON_PROFILER_

#include <chrono>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "cuda_runtime_api.h"
#include "trt_converter/common/device.h"

namespace sss {
class TimeScopedProfiler {
 public:
  TimeScopedProfiler(std::string_view prefix_str) : prefix_str_(prefix_str) {
    start_time_ = std::chrono::high_resolution_clock::now();
  }
  ~TimeScopedProfiler() {
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start_time_).count();
    std::cout << prefix_str_ << " : "
              << " elapsed time is  " << elapsed_time << std::endl;
  }

 private:
  std::chrono::system_clock::time_point start_time_;
  std::string_view prefix_str_;
};

struct LayerProfiler {
  std::string layer_name;
  float ms;
};

class Profiler : public nvinfer1::IProfiler {
 public:
  void reportLayerTime(const char* layerName, float ms) noexcept override;
  void Print(std::ostream& os);
  void ExportJSONProfile(const std::string& fileName) const;

 private:
  float GetTotalTime() const;
  std::vector<LayerProfiler> layers_;
  std::vector<LayerProfiler>::iterator l_iterator_ = layers_.begin();
  int update_count_;
};

class TimerBase {
 public:
  virtual void Start();
  virtual void Stop();
  float Microseconds() const noexcept { return ms_ * 1000.f; }
  float Milliseconds() const noexcept { return ms_; }
  float Seconds() const noexcept { return ms_ / 1000.f; }
  void Reset() noexcept { ms_ = 0.f; }

 protected:
  float ms_{0.0f};
};
class GpuTimer : public TimerBase {
 public:
  GpuTimer(cudaStream_t stream) : stream_(stream) {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&end_));
  }
  ~GpuTimer();
  void Start() override;
  void Stop() override;

 private:
  cudaEvent_t start_;
  cudaEvent_t end_;
  cudaStream_t stream_;
};

class CpuTimer : public TimerBase {
 public:
  void Start() override;
  void Stop() override;

 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
  std::chrono::time_point<std::chrono::high_resolution_clock> end_;
};

}  // namespace sss

#endif /* INCLUDE_TRT_CONVERTER_COMMON_PROFILER_ */
