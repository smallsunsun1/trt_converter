#include "trt_converter/common/profiler.h"

#include <algorithm>
#include <numeric>

namespace sss {

float Profiler::GetTotalTime() const {
  float time = std::accumulate(layers_.begin(), layers_.end(), 0, [](float l1, const LayerProfiler& l2) { return l1 + l2.ms; });
  return time;
}

}  // namespace sss