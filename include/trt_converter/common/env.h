#ifndef INCLUDE_TRT_CONVERTER_COMMON_ENV_
#define INCLUDE_TRT_CONVERTER_COMMON_ENV_

#include <memory>
#include <vector>

#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "trt_converter/common/options.h"
#include "trt_converter/common/profiler.h"
#include "trt_converter/common/utils.h"

namespace sss {
struct InferenceOptions;

struct TRTInferenceEnvironment {
  TRTUniquePtr<nvinfer1::ICudaEngine> engine;
  std::unique_ptr<nvinfer1::IProfiler> profiler;
  std::vector<TRTUniquePtr<nvinfer1::IExecutionContext>> contexts;
  std::vector<std::unique_ptr<Bindings>> bindings;
};

bool SetupInference(TRTInferenceEnvironment& env, const InferenceOptions& options);
void RunInference(const InferenceOptions& options, TRTInferenceEnvironment& iEnv, int device);

}  // namespace sss

#endif /* INCLUDE_TRT_CONVERTER_COMMON_ENV_ */
