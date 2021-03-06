#include "trt_converter/common/logging.h"

#include <glog/logging.h>
#include <ostream>

namespace sss {

void Logger::log(Severity severity, const char* msg) TRTC_NOEXCEPT {
  switch (severity) {
    case Severity::kINFO:
      LOG(INFO) << msg;
      break;
    case Severity::kERROR:
      LOG(ERROR) << msg;
      break;
    case Severity::kWARNING:
      LOG(WARNING) << msg;
      break;
    case Severity::kINTERNAL_ERROR:
      LOG(ERROR) << msg;
    case Severity::kVERBOSE:
      LOG(INFO) << msg;
  }
}

Logger& Logger::Instance() {
  static Logger logger;
  return logger;
}

}  // namespace sss