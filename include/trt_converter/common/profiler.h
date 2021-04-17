#ifndef INCLUDE_TRT_CONVERTER_COMMON_PROFILER_
#define INCLUDE_TRT_CONVERTER_COMMON_PROFILER_

#include <chrono>
#include <iostream>
#include <string_view>

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

class CpuTimeProfiler: public BaseProfiler {};

}  // namespace sss

#endif /* INCLUDE_TRT_CONVERTER_COMMON_PROFILER_ */
