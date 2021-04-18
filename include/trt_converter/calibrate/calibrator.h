#ifndef INCLUDE_TRT_CONVERTER_CALIBRATE_CALIBRATOR_
#define INCLUDE_TRT_CONVERTER_CALIBRATE_CALIBRATOR_

#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "trt_converter/common/macro.h"

namespace sss {

template <typename Stream>
class EntropyCalibratorImpl {
 public:
  EntropyCalibratorImpl(Stream stream, uint32_t first_batch, std::string network_name, const char* input_blob_name, bool read_cache = true)
      : stream_(std::move(stream)),
        calibrate_string_name_("CalibrationTable" + network_name),
        input_blob_name_(input_blob_name),
        read_cache_(read_cache) {
    nvinfer1::Dims dim = stream_->GetDims();
    uint32_t res = 1;
    for (uint32_t i = 0; i < dim.nbDims; ++i) {
      res *= dim.d[i];
    }
    input_count_ = res;
    CUDA_CHECK(cudaMalloc(&device_input_, input_count_));
    stream_.Reset(first_batch);
  }
  int getBatchSize() const { return stream_.GetBatchSize(); }
  bool getBatch(void* bindings[], const char* names[], int nb_bindings) {
    if (!stream_.Next()) {
      return false;
    }
    CUDA_CHECK(cudaMemCpy(device_input_, stream_.GetBatch().get(), input_count_ * sizeof(float), cudaMemcpyHostToDevice));
    assert(!strcmp(names[0], input_blob_name_.data()));
    bindings[0] = device_input_;
    return true;
  }
  virtual ~EntropyCalibratorImpl() { CUDA_CHECK(cudaFree(device_input_)); }
  const void* readCalibrationCache(size_t& length) {
    std::ifstream file_data(calibrate_string_name_, std::ios::binary);
    if (read_cache_ && file_data.good()) {
      std::copy(std::istreambuf_iterator<char>(file_data), std::istreambuf_iterator<char>(), std::back_inserter(calibration_cache_));
    }
    length = calibration_cache_.size();
    return length ? calibration_cache_.data() : nullptr;
  }
  void writeCalibrationCache(const void* cache, size_t length) {
    std::ofstream file_data(calibrate_string_name_, std::ios::binary | std::ios::out);
    file_data.write(reinterpret_cast<const char*>(cache), length);
  }

 private:
  Stream stream_;
  uint32_t input_count_ = 0;
  std::string calibrate_string_name_;
  std::string_view input_blob_name_;
  bool read_cache_ = true;
  void* device_input_ = nullptr;
  std::vector<char> calibration_cache_;
};

template <typename Stream>
class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2 {
 public:
  Int8EntropyCalibrator(Stream stream, int first_batch, const char* network_name, const char* input_blob_name, bool read_cache = true)
      : calib_impl_(stream, first_batch, network_name, input_blob_name, read_cache) {}

  int getBatchSize() const override { return calib_impl_.getBatchSize(); }

  bool getBatch(void* bindings[], const char* names[], int nbBindings) override { return calib_impl_.getBatch(bindings, names, nbBindings); }

  const void* readCalibrationCache(size_t& length) override { return calib_impl_.readCalibrationCache(length); }

  void writeCalibrationCache(const void* cache, size_t length) override { calib_impl_.writeCalibrationCache(cache, length); }

 private:
  EntropyCalibratorImpl<Stream> calib_impl_;
};

}  // namespace sss

#endif /* INCLUDE_TRT_CONVERTER_CALIBRATE_CALIBRATOR_ */
