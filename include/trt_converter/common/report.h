#ifndef INCLUDE_TRT_CONVERTER_COMMON_REPORT_
#define INCLUDE_TRT_CONVERTER_COMMON_REPORT_

#include <NvInfer.h>
#include <iostream>
#include <string>
#include <vector>

#include "trt_converter/common/options.h"
#include "trt_converter/common/utils.h"

namespace nvinfer1 {
class IExecutionContext;
}  // namespace nvinfer1

namespace sss {
class Bindings;
struct ReportingOptions;

//!
//! \struct InferenceTime
//! \brief Measurement times in milliseconds
//!
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
  float latency() const { return in + compute + out; }
};

//!
//! \struct InferenceTrace
//! \brief Measurement points in milliseconds
//!
struct InferenceTrace {
  InferenceTrace(int s, float es, float ee, float is, float ie, float cs, float ce, float os, float oe)
      : stream(s),
        enq_start(es),
        enq_end(ee),
        in_start(is),
        in_end(ie),
        compute_start(cs),
        compute_end(ce),
        out_start(os),
        out_end(oe) {}

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

//!
//! \brief Print benchmarking time and number of traces collected
//!
void PrintProlog(int warmups, int timings, float warmupMs, float walltime, std::ostream& os);

//!
//! \brief Print a timing trace
//!
void PrintTiming(const std::vector<InferenceTime>& timings, int runsPerAvg, std::ostream& os);

//!
//! \brief Print the performance summary of a trace
//!
void PrintEpilog(std::vector<InferenceTime> timings, float percentile, int queries, std::ostream& os);

//!
//! \brief Print and summarize a timing trace
//!
void PrintPerformanceReport(const std::vector<InferenceTrace>& trace, const ReportingOptions& reporting, float warmupMs,
                            int queries, std::ostream& os);

//!
//! \brief Export a timing trace to JSON file
//!
void ExportJSONTrace(const std::vector<InferenceTrace>& trace, const std::string& fileName);

//!
//! \brief Print input tensors to stream
//!
void DumpInputs(const nvinfer1::IExecutionContext& context, const Bindings& bindings, std::ostream& os);

//!
//! \brief Print output tensors to stream
//!
void DumpOutputs(const nvinfer1::IExecutionContext& context, const Bindings& bindings, std::ostream& os);

//!
//! \brief Export output tensors to JSON file
//!
void ExportJSONOutput(const nvinfer1::IExecutionContext& context, const Bindings& bindings,
                      const std::string& fileName);

}  // namespace sss

#endif /* INCLUDE_TRT_CONVERTER_COMMON_REPORT_ */
