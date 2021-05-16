#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "cuda_runtime_api.h"
#include "trt_converter/calibrate/calibrator.h"
#include "trt_converter/calibrate/stream.h"
#include "trt_converter/common/argparser.h"
#include "trt_converter/common/buffer.h"
#include "trt_converter/common/common.h"
#include "trt_converter/common/logging.h"
#include "trt_converter/common/utils.h"

const std::string kSampleName = "TensorRT.sample_int8";

namespace sss {
struct SampleInt8Params : public CaffeModelParams {
  uint32_t nb_cal_batches;
  uint32_t cal_batch_size;
  std::string network_name;
};

class SampleInt8 {
 public:
  template <typename T>
  using SampleUniquePtr = std::unique_ptr<T, InferDeleter>;
  SampleInt8(const SampleInt8Params& params) : params_(params), engine_(nullptr) {
    initLibNvInferPlugins(Logger::Instance(), "");
  }
  bool Build(nvinfer1::DataType dataType);

  bool IsSupported(nvinfer1::DataType dataType);

  bool Infer(std::vector<float>& score, int firstScoreBatch, int nbScoreBatches);

  bool Teardown();

 private:
  SampleInt8Params params_;
  nvinfer1::Dims input_dims_;
  std::shared_ptr<nvinfer1::ICudaEngine> engine_;

  bool ConstructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
                        SampleUniquePtr<nvinfer1::INetworkDefinition>& network,
                        SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
                        SampleUniquePtr<nvcaffeparser1::ICaffeParser>& parser, nvinfer1::DataType dataType);

  bool ProcessInput(const BufferManager& buffers, const float* data);

  int CalculateScore(const BufferManager& buffers, float* labels, int batchSize, int outputSize, int threshold);
};

bool SampleInt8::Build(nvinfer1::DataType dataType) {
  return true;
}

bool SampleInt8::IsSupported(nvinfer1::DataType dataType) {
  return true;
}

bool SampleInt8::Teardown() {
  return true;
}

bool SampleInt8::Infer(std::vector<float> &score, int firstScoreBatch, int nbScoreBatches) {
  return true;
}

}  // namespace sss
