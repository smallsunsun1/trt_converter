#include "trt_converter/common/logging.h"

#include <cassert>

namespace sss {

void MessageBuffer::AddMessage(const std::string &message) {
  std::lock_guard<std::mutex> lock(mutex);
  message_data.push(message);
}

std::optional<std::string> MessageBuffer::GetMessage() {
  std::lock_guard<std::mutex> lock(mutex);
  if (!message_data.empty()) {
    auto res = std::make_optional(message_data.front());
    message_data.pop();
    return res;
  } else {
    return std::optional<std::string>();
  }
}

void MessageConsumer::SetShouldReport(Severity severity) { should_log_ = (severity_ <= severity); }

std::ostream &MessageConsumer::GetStreamBySeverity(Severity severity) {
  if (severity >= Severity::kINFO)
    return std::cout;
  else {
    return std::cerr;
  }
}

std::string MessageConsumer::GetPrefixString(Severity severity) {
  switch (severity) {
    case Severity::kINTERNAL_ERROR:
      return "[F] ";
    case Severity::kERROR:
      return "[E] ";
    case Severity::kWARNING:
      return "[W] ";
    case Severity::kINFO:
      return "[I] ";
    case Severity::kVERBOSE:
      return "[V] ";
    default:
      assert(0);
      return "";
  }
}

void Logger::log(Severity severity, const char *msg) TRTC_NOEXCEPT {}

}  // namespace sss