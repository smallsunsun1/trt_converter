#ifndef INCLUDE_TRT_CONVERTER_COMMON_PROFILER_
#define INCLUDE_TRT_CONVERTER_COMMON_PROFILER_

#include <chrono>
#include <iostream>
#include <string_view>
#include <vector>

#include "NvInfer.h"

namespace sss {
class TimeScopedProfiler {
 public:
  TimeScopedProfiler(std::string_view prefix_str) : prefix_str_(prefix_str) { start_time_ = std::chrono::high_resolution_clock::now(); }
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

class BaseProfiler {};

class CpuTimeProfiler : public BaseProfiler {};

struct InferenceTime {
  InferenceTime(float q, float i, float c, float o, float e) : enq(q), in(i), compute(c), out(o), e2e(e) {}

  InferenceTime() = default;
  InferenceTime(const InferenceTime&) = default;
  InferenceTime(InferenceTime&&) = default;
  InferenceTime& operator=(const InferenceTime&) = default;
  InferenceTime& operator=(InferenceTime&&) = default;
  ~InferenceTime() = default;

  float enq{0};      // Enqueue
  float in{0};       // Host to Device
  float compute{0};  // Compute
  float out{0};      // Device to Host
  float e2e{0};      // end to end

  // ideal latency
  float Latency() const { return in + compute + out; }
};

struct InferenceTrace {
  InferenceTrace(int s, float es, float ee, float is, float ie, float cs, float ce, float os, float oe)
      : stream(s), enq_start(es), enq_end(ee), in_start(is), in_end(ie), compute_start(cs), compute_end(ce), out_start(os), out_end(oe) {}

  InferenceTrace() = default;
  InferenceTrace(const InferenceTrace&) = default;
  InferenceTrace(InferenceTrace&&) = default;
  InferenceTrace& operator=(const InferenceTrace&) = default;
  InferenceTrace& operator=(InferenceTrace&&) = default;
  ~InferenceTrace() = default;

  int stream{0};
  float enq_start{0};
  float enq_end{0};
  float in_start{0};
  float in_end{0};
  float compute_start{0};
  float compute_end{0};
  float out_start{0};
  float out_end{0};
};

inline InferenceTime operator+(const InferenceTime& a, const InferenceTime& b) {
  return InferenceTime(a.enq + b.enq, a.in + b.in, a.compute + b.compute, a.out + b.out, a.e2e + b.e2e);
}

inline InferenceTime operator+=(InferenceTime& a, const InferenceTime& b) { return a = a + b; }

struct LayerProfiler {
  std::string layer_name;
  float ms;
};

class Profiler : public nvinfer1::IProfiler {
 public:
  void reportLayerTime(const char* layerName, float ms) noexcept override;
  void Print(std::ostream& os);

 private:
  float GetTotalTime() const;
  std::vector<LayerProfiler> layers_;
  int update_count_;
};

}  // namespace sss

#endif /* INCLUDE_TRT_CONVERTER_COMMON_PROFILER_ */
