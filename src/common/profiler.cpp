#include "trt_converter/common/profiler.h"

#include <algorithm>
#include <iomanip>
#include <numeric>

#include "trt_converter/common/report.h"

namespace sss {

float Profiler::GetTotalTime() const {
  float time =
      std::accumulate(layers_.begin(), layers_.end(), 0, [](float l1, const LayerProfiler& l2) { return l1 + l2.ms; });
  return time;
}

void Profiler::reportLayerTime(const char* layerName, float ms) noexcept {
  auto l_iterator_ = layers_.begin();
  if (l_iterator_ == layers_.end()) {
    const bool first = !layers_.empty() && layers_.begin()->layer_name == layerName;
    update_count_ += layers_.empty() || first;
    if (first) {
      l_iterator_ = layers_.begin();
      {
        layers_.emplace_back();
        layers_.back().layer_name = layerName;
        l_iterator_ = layers_.end() - 1;
      }
      l_iterator_->ms += ms;
      ++l_iterator_;
    }
  }
}

void Profiler::Print(std::ostream& os) {
  const std::string name_hdr("Layer");
  const std::string time_hdr("   Time (ms)");
  const std::string avg_hdr("   Avg. Time (ms)");
  const std::string percentage_hdr("   Time \%");

  const float total_time_ms = GetTotalTime();

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
       << std::setw(percentage_length) << std::fixed << std::setprecision(1) << p.ms / total_time_ms * 100 << std::endl;
  }
  {
    os << std::setw(name_length) << "Total" << std::setw(time_length) << std::fixed << std::setprecision(2)
       << total_time_ms << std::setw(avg_length) << std::fixed << std::setprecision(4) << total_time_ms / update_count_
       << std::setw(percentage_length) << std::fixed << std::setprecision(1) << 100.0 << std::endl;
    // clang on
  }
  os << std::endl;
}

void Profiler::ExportJSONProfile(const std::string& filename) const {
  std::ofstream os(filename, std::ofstream::trunc);
  os << "[" << std::endl << "  { \"count\" : " << update_count_ << " }" << std::endl;

  const auto totalTimeMs = GetTotalTime();

  for (const auto& l : layers_) {
    // clang off
    os << ", {"
       << " \"name\" : \"" << l.layer_name
       << "\""
          ", \"timeMs\" : "
       << l.ms << ", \"averageMs\" : " << l.ms / update_count_ << ", \"percentage\" : " << l.ms / totalTimeMs * 100
       << " }" << std::endl;
    // clang on
  }
  os << "]" << std::endl;
}

void DumpInputs(const nvinfer1::IExecutionContext& context, const Bindings& bindings, std::ostream& os) {
  os << "Input Tensors:" << std::endl;
  bindings.DumpInputs(context, os);
}

void DumpOutputs(const nvinfer1::IExecutionContext& context, const Bindings& bindings, std::ostream& os) {
  os << "Output Tensors:" << std::endl;
  bindings.DumpOutputs(context, os);
}

void ExportJSONOutput(const nvinfer1::IExecutionContext& context, const Bindings& bindings,
                      const std::string& filename) {
  std::ofstream os(filename, std::ofstream::trunc);
  std::string sep = "  ";
  const auto output = bindings.GetOutputBindings();
  os << "[" << std::endl;
  for (const auto& binding : output) {
    // clang off
    os << sep << "{ \"name\" : \"" << binding.first << "\"" << std::endl;
    sep = ", ";
    os << "  " << sep << "\"dimensions\" : \"";
    bindings.DumpBindingDimensions(binding.second, context, os);
    os << "\"" << std::endl;
    os << "  " << sep << "\"values\" : [ ";
    bindings.DumpBindingValues(binding.second, os, sep);
    os << " ]" << std::endl << "  }" << std::endl;
    // clang on
  }
  os << "]" << std::endl;
}

}  // namespace sss