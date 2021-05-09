#include "trt_converter/common/report.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <utility>

namespace sss {
namespace {

template <typename T>
float FindPercentile(float percentage, const std::vector<InferenceTime>& timings, const T& to_float) {
  const int all = static_cast<int>(timings.size());
  const int exclude = static_cast<int>((1 - percentage / 100) * all);
  if (0 <= exclude && exclude <= all) {
    return to_float(timings[std::max(all - 1 - exclude, 0)]);
  }
  return std::numeric_limits<float>::infinity();
}

template <typename T>
float FindMedian(const std::vector<InferenceTime>& timings, const T& to_float) {
  if (timings.empty()) {
    return std::numeric_limits<float>::infinity();
  }

  const int m = timings.size() / 2;
  if (timings.size() % 2) {
    return to_float(timings[m]);
  }

  return (to_float(timings[m - 1]) + to_float(timings[m])) / 2;
}

inline InferenceTime TraceToTiming(const InferenceTrace& trace) {
  return InferenceTime((trace.enq_end - trace.enq_start), (trace.in_end - trace.in_start),
                       (trace.compute_end - trace.compute_start), (trace.out_end - trace.out_start),
                       (trace.out_end - trace.in_start));
};

}  // namespace

void PrintProlog(int warmups, int timings, float warmupMs, float benchTimeMs, std::ostream& os) {
  os << "Warmup completed " << warmups << " queries over " << warmupMs << " ms" << std::endl;
  os << "Timing trace has " << timings << " queries over " << benchTimeMs / 1000 << " s" << std::endl;
}

void PrintTiming(const std::vector<InferenceTime>& timings, int runs_per_avg, std::ostream& os) {
  int count = 0;
  InferenceTime sum;

  os << "Trace averages of " << runs_per_avg << " runs:" << std::endl;
  for (const auto& t : timings) {
    sum += t;

    if (++count == runs_per_avg) {
      // clang off
      os << "Average on " << runs_per_avg << " runs - GPU latency: " << sum.compute / runs_per_avg
         << " ms - Host latency: " << sum.latency() / runs_per_avg << " ms (end to end " << sum.e2e / runs_per_avg
         << " ms, enqueue " << sum.enq / runs_per_avg << " ms)" << std::endl;
      // clang on
      count = 0;
      sum.enq = 0;
      sum.in = 0;
      sum.compute = 0;
      sum.out = 0;
      sum.e2e = 0;
    }
  }
}

void PrintEpilog(std::vector<InferenceTime> timings, float walltimeMs, float percentile, int queries,
                 std::ostream& os) {
  const InferenceTime total_time = std::accumulate(timings.begin(), timings.end(), InferenceTime());

  const auto get_latency = [](const InferenceTime& t) { return t.latency(); };
  const auto cmp_latency = [](const InferenceTime& a, const InferenceTime& b) { return a.latency() < b.latency(); };
  std::sort(timings.begin(), timings.end(), cmp_latency);
  const float latency_min = timings.front().latency();
  const float latency_max = timings.back().latency();
  const float latency_median = FindMedian(timings, get_latency);
  const float latency_percentile = FindPercentile(percentile, timings, get_latency);
  const float latency_throughput = queries * timings.size() / walltimeMs * 1000;

  const auto get_end_to_end = [](const InferenceTime& t) { return t.e2e; };
  const auto cmp_end_to_end = [](const InferenceTime& a, const InferenceTime& b) { return a.e2e < b.e2e; };
  std::sort(timings.begin(), timings.end(), cmp_end_to_end);
  const float end_to_end_min = timings.front().e2e;
  const float end_to_end_max = timings.back().e2e;
  const float end_to_end_median = FindMedian(timings, get_end_to_end);
  const float end_to_end_percentile = FindPercentile(percentile, timings, get_end_to_end);

  const auto get_compute = [](const InferenceTime& t) { return t.compute; };
  const auto cmp_compute = [](const InferenceTime& a, const InferenceTime& b) { return a.compute < b.compute; };
  std::sort(timings.begin(), timings.end(), cmp_compute);
  const float gpu_min = timings.front().compute;
  const float gpu_max = timings.back().compute;
  const float gpu_median = FindMedian(timings, get_compute);
  const float gpu_percentile = FindPercentile(percentile, timings, get_compute);

  const auto get_enqueue = [](const InferenceTime& t) { return t.enq; };
  const auto cmp_enqueue = [](const InferenceTime& a, const InferenceTime& b) { return a.enq < b.enq; };
  std::sort(timings.begin(), timings.end(), cmp_enqueue);
  const float enq_min = timings.front().enq;
  const float enq_max = timings.back().enq;
  const float enq_median = FindMedian(timings, get_enqueue);

  // clang off
  os << "Host Latency" << std::endl
     << "min: " << latency_min
     << " ms "
        "(end to end "
     << end_to_end_min << " ms)" << std::endl
     << "max: " << latency_max
     << " ms "
        "(end to end "
     << end_to_end_max << " ms)" << std::endl
     << "mean: " << total_time.latency() / timings.size()
     << " ms "
        "(end to end "
     << total_time.e2e / timings.size() << " ms)" << std::endl
     << "median: " << latency_median
     << " ms "
        "(end to end "
     << end_to_end_median << " ms)" << std::endl
     << "percentile: " << latency_percentile
     << " ms "
        "at "
     << percentile
     << "% "
        "(end to end "
     << end_to_end_percentile
     << " ms "
        "at "
     << percentile << "%)" << std::endl
     << "throughput: " << latency_throughput << " qps" << std::endl
     << "walltime: " << walltimeMs / 1000 << " s" << std::endl
     << "Enqueue Time" << std::endl
     << "min: " << enq_min << " ms" << std::endl
     << "max: " << enq_max << " ms" << std::endl
     << "median: " << enq_median << " ms" << std::endl
     << "GPU Compute" << std::endl
     << "min: " << gpu_min << " ms" << std::endl
     << "max: " << gpu_max << " ms" << std::endl
     << "mean: " << total_time.compute / timings.size() << " ms" << std::endl
     << "median: " << gpu_median << " ms" << std::endl
     << "percentile: " << gpu_percentile
     << " ms "
        "at "
     << percentile << "%" << std::endl
     << "total compute time: " << total_time.compute / 1000 << " s" << std::endl;
  // clang on
}

void PrintPerformanceReport(const std::vector<InferenceTrace>& trace, const ReportingOptions& reporting,
                            float warmup_ms, int queries, std::ostream& os) {
  const auto is_not_warmup = [&warmup_ms](const InferenceTrace& a) { return a.compute_start >= warmup_ms; };
  const auto no_warmup = std::find_if(trace.begin(), trace.end(), is_not_warmup);
  const int warmups = no_warmup - trace.begin();
  const float bench_time = trace.back().out_end - no_warmup->in_start;
  PrintProlog(warmups * queries, (trace.size() - warmups) * queries, warmup_ms, bench_time, os);

  std::vector<InferenceTime> timings(trace.size() - warmups);
  std::transform(no_warmup, trace.end(), timings.begin(), TraceToTiming);
  PrintTiming(timings, reporting.avgs, os);
  PrintEpilog(timings, bench_time, reporting.percentile, queries, os);

  if (!reporting.export_times.empty()) {
    ExportJSONTrace(trace, reporting.export_times);
  }
}

void ExportJSONTrace(const std::vector<InferenceTrace>& trace, const std::string& filename) {
  std::ofstream os(filename, std::ofstream::trunc);
  os << "[" << std::endl;
  const char* sep = "  ";
  for (const auto& t : trace) {
    const InferenceTime it(TraceToTiming(t));
    os << sep << "{ ";
    sep = ", ";
    // clang off
    os << "\"startEnqMs\" : " << t.in_start << sep << "\"endEnqMs\" : " << t.in_end << sep
       << "\"startInMs\" : " << t.enq_start << sep << "\"endInMs\" : " << t.enq_end << sep
       << "\"startComputeMs\" : " << t.compute_start << sep << "\"endComputeMs\" : " << t.compute_end << sep
       << "\"startOutMs\" : " << t.out_start << sep << "\"endOutMs\" : " << t.out_end << sep << "\"inMs\" : " << it.in
       << sep << "\"computeMs\" : " << it.compute << sep << "\"outMs\" : " << it.out << sep
       << "\"latencyMs\" : " << it.latency() << sep << "\"endToEndMs\" : " << it.e2e << " }" << std::endl;
    // clang on
  }
  os << "]" << std::endl;
}

}  // namespace sss