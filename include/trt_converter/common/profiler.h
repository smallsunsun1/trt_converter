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

}  // namespace sss

#endif /* INCLUDE_TRT_CONVERTER_COMMON_PROFILER_ */
