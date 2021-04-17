#ifndef INCLUDE_TRT_CONVERTER_COMMON_LOGGING_
#define INCLUDE_TRT_CONVERTER_COMMON_LOGGING_

#include <iostream>
#include <mutex>
#include <optional>
#include <queue>
#include <string>

#include "NvInferRuntimeCommon.h"
#include "macro.h"

namespace sss {

using Severity = nvinfer1::ILogger::Severity;

struct MessageBuffer {
  void AddMessage(const std::string& message);
  std::optional<std::string> GetMessage();
  std::queue<std::string> message_data;
  std::mutex mutex;
};

class MessageConsumer {
 public:
  void SetShouldReport(Severity severity);
  static std::ostream& GetStreamBySeverity(Severity severity);
  static std::string GetPrefixString(Severity severity);

 private:
  bool should_log_;
  Severity severity_;
  MessageBuffer message_buffer_;
};

class Logger : public nvinfer1::ILogger {
 public:
  virtual void log(Severity severity, const char* msg) TRTC_NOEXCEPT override;

 private:
  MessageBuffer buffer_;
};

}  // namespace sss

#endif /* INCLUDE_TRT_CONVERTER_COMMON_LOGGING_ */
