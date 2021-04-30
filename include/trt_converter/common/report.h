#ifndef INCLUDE_TRT_CONVERTER_COMMON_REPORT_
#define INCLUDE_TRT_CONVERTER_COMMON_REPORT_

#include <NvInfer.h>

#include <iostream>
#include <vector>

#include "trt_converter/common/options.h"
#include "trt_converter/common/utils.h"

namespace sss {

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
      : stream(s), enqStart(es), enqEnd(ee), inStart(is), inEnd(ie), computeStart(cs), computeEnd(ce), outStart(os), outEnd(oe) {}

  InferenceTrace() = default;
  InferenceTrace(const InferenceTrace&) = default;
  InferenceTrace(InferenceTrace&&) = default;
  InferenceTrace& operator=(const InferenceTrace&) = default;
  InferenceTrace& operator=(InferenceTrace&&) = default;
  ~InferenceTrace() = default;

  int stream{0};
  float enqStart{0};
  float enqEnd{0};
  float inStart{0};
  float inEnd{0};
  float computeStart{0};
  float computeEnd{0};
  float outStart{0};
  float outEnd{0};
};

inline InferenceTime operator+(const InferenceTime& a, const InferenceTime& b) {
  return InferenceTime(a.enq + b.enq, a.in + b.in, a.compute + b.compute, a.out + b.out, a.e2e + b.e2e);
}

inline InferenceTime operator+=(InferenceTime& a, const InferenceTime& b) { return a = a + b; }

//!
//! \brief Print benchmarking time and number of traces collected
//!
void printProlog(int warmups, int timings, float warmupMs, float walltime, std::ostream& os);

//!
//! \brief Print a timing trace
//!
void printTiming(const std::vector<InferenceTime>& timings, int runsPerAvg, std::ostream& os);

//!
//! \brief Print the performance summary of a trace
//!
void printEpilog(std::vector<InferenceTime> timings, float percentile, int queries, std::ostream& os);

//!
//! \brief Print and summarize a timing trace
//!
void printPerformanceReport(const std::vector<InferenceTrace>& trace, const ReportingOptions& reporting, float warmupMs, int queries,
                            std::ostream& os);

//!
//! \brief Export a timing trace to JSON file
//!
void exportJSONTrace(const std::vector<InferenceTrace>& trace, const std::string& fileName);

//!
//! \brief Print input tensors to stream
//!
void dumpInputs(const nvinfer1::IExecutionContext& context, const Bindings& bindings, std::ostream& os);

//!
//! \brief Print output tensors to stream
//!
void dumpOutputs(const nvinfer1::IExecutionContext& context, const Bindings& bindings, std::ostream& os);

//!
//! \brief Export output tensors to JSON file
//!
void exportJSONOutput(const nvinfer1::IExecutionContext& context, const Bindings& bindings, const std::string& fileName);

}  // namespace sss

#endif /* INCLUDE_TRT_CONVERTER_COMMON_REPORT_ */
