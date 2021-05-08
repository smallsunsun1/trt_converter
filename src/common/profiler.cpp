#include "trt_converter/common/profiler.h"

#include <algorithm>
#include <iomanip>
#include <numeric>

namespace sss {

float Profiler::GetTotalTime() const {
  float time =
      std::accumulate(layers_.begin(), layers_.end(), 0, [](float l1, const LayerProfiler& l2) { return l1 + l2.ms; });
  return time;
}

void Profiler::Print(std::ostream& os) {
  const std::string name_hdr("Layer");
  const std::string time_hdr("   Time (ms)");
  const std::string avg_hdr("   Avg. Time (ms)");
  const std::string percentage_hdr("   Time \%");

  const float totalTimeMs = GetTotalTime();

  const auto cmp_layer = [](const LayerProfiler& a, const LayerProfiler& b) {
    return a.layer_name.size() < b.layer_name.size();
  };
  const auto longest_name = std::max_element(layers_.begin(), layers_.end(), cmp_layer);
  const auto name_length = std::max(longest_name->layer_name.size() + 1, name_hdr.size());
  const auto time_length = time_hdr.size();
  const auto avg_length = avg_hdr.size();
  const auto percentage_length = percentage_hdr.size();

  os << std::endl
     << "=== Profile (" << update_count_ << " iterations ) ===" << std::endl
     << std::setw(name_length) << name_hdr << time_hdr << avg_hdr << percentage_hdr << std::endl;

  for (const auto& p : layers_) {
    // clang off
    os << std::setw(name_length) << p.layer_name << std::setw(time_length) << std::fixed << std::setprecision(2) << p.ms
       << std::setw(avg_length) << std::fixed << std::setprecision(4) << p.ms / update_count_
       << std::setw(percentage_length) << std::fixed << std::setprecision(1) << p.ms / totalTimeMs * 100 << std::endl;
  }
  {
    os << std::setw(name_length) << "Total" << std::setw(time_length) << std::fixed << std::setprecision(2)
       << totalTimeMs << std::setw(avg_length) << std::fixed << std::setprecision(4) << totalTimeMs / update_count_
       << std::setw(percentage_length) << std::fixed << std::setprecision(1) << 100.0 << std::endl;
    // clang on
  }
  os << std::endl;
}

}  // namespace sss