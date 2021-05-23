#include <memory>

#include "trt_converter/common/logging.h"

using namespace sss;

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  (void)argc;
  std::unique_ptr<Logger> logger = std::make_unique<Logger>();
  logger->log(Severity::kINFO, "info test");
  logger->log(Severity::kERROR, "error test");
  google::ShutdownGoogleLogging();
  return 0;
}