#include "trt_converter/common/env.h"

namespace sss {

bool SetupInference(TRTInferenceEnvironment &env, const InferenceOptions &options) {
  for (uint32_t i = 0; i < options.streams; ++i) {
    env.contexts.emplace_back(env.engine->createExecutionContext());
    env.bindings_.emplace_back(new Bindings);
  }
  if (env.profiler) {
    env.contexts.front()->setProfiler(env.profiler.get());
  }
  const int nOptProfiles = env.engine->getNbOptimizationProfiles();
  const int nBindings = env.engine->getNbBindings();
  const int bindingsInProfile = nOptProfiles > 0 ? nBindings / nOptProfiles : 0;
  const int endBindingIndex = bindingsInProfile ? bindingsInProfile : env.engine->getNbBindings();
}

}  // namespace sss