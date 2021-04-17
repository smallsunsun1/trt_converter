#ifndef INCLUDE_TRT_CONVERTER_COMMON_ENV_
#define INCLUDE_TRT_CONVERTER_COMMON_ENV_

#include <memory>
#include <vector>
#include "NvInfer.h"
#include "profiler.h"

namespace sss {

template <typename T>
struct TRTObjDeleter {
    void operator()(T* obj) {
        obj->destroy();
    }
};

template <typename T>
using TRTUniquePtr = std::unique_ptr<T, TRTObjDeleter<T>>;

struct TRTInferenceEnvironment {
    TRTUniquePtr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<BaseProfiler> profiler;
    std::vector<TRTUniquePtr<nvinfer1::IExecutionContext>> contexts;
};

}

#endif /* INCLUDE_TRT_CONVERTER_COMMON_ENV_ */
