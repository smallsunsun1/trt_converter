#ifndef INCLUDE_TRT_CONVERTER_COMMON_LOGGING_
#define INCLUDE_TRT_CONVERTER_COMMON_LOGGING_

#include "trt_converter/common/macro.h"

#include "NvInferRuntimeCommon.h"
#include "glog/logging.h"

namespace sss {

using Severity = nvinfer1::ILogger::Severity;

class Logger : public nvinfer1::ILogger {
 public:
  virtual void log(Severity severity, const char* msg) TRTC_NOEXCEPT override;
};

}  // namespace sss

#endif /* INCLUDE_TRT_CONVERTER_COMMON_LOGGING_ */
