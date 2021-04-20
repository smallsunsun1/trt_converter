#ifndef INCLUDE_TRT_CONVERTER_COMMON_ENV_
#define INCLUDE_TRT_CONVERTER_COMMON_ENV_

#include <memory>
#include <vector>

#include "NvInfer.h"
#include "profiler.h"
#include "utils.h"
namespace sss {



struct TRTInferenceEnvironment {
  TRTUniquePtr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<BaseProfiler> profiler;
  std::vector<TRTUniquePtr<nvinfer1::IExecutionContext>> contexts;
  std::vector<std::unique_ptr<Bindings>> bindings_;
};

}  // namespace sss

#endif /* INCLUDE_TRT_CONVERTER_COMMON_ENV_ */
