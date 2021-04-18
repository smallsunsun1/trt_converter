#ifndef INCLUDE_TRT_CONVERTER_CALIBRATE_STREAM_
#define INCLUDE_TRT_CONVERTER_CALIBRATE_STREAM_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "NvInfer.h"

namespace sss {

class DataStreamBase {
 public:
  virtual void Reset(uint32_t first_batch) = 0;
  virtual bool Next() = 0;
  virtual void Skip(uint32_t count) = 0;
  virtual std::unique_ptr<float[]> GetBatch() = 0;
  virtual uint32_t GetBatchesRead() const = 0;
  virtual uint32_t GetBatchSize() const = 0;
  virtual nvinfer1::Dims GetDims() const = 0;
};

//
class ImageDataStream : public DataStreamBase {
 public:
  ImageDataStream(uint32_t batch_size, nvinfer1::Dims dims, std::vector<std::string> image_filenames)
      : batch_size_(batch_size), dims_(std::move(dims)), image_filenames_(std::move(image_filenames)) {}
  virtual void Reset(uint32_t first_batch) override {
    current_batch = 0;
    Skip(first_batch);
  }
  virtual bool Next() override {
    if (current_batch >= image_filenames_.size()) {
      return false;
    }
    current_batch++;
    return false;
  }
  virtual void Skip(uint32_t count) override { current_batch += count; }
  virtual std::unique_ptr<float[]> GetBatch() override;
  virtual uint32_t GetBatchSize() const override { return batch_size_; }
  virtual uint32_t GetBatchesRead() const override;
  virtual nvinfer1::Dims GetDims() const override { return dims_; }

 private:
  std::vector<float> ReadImageData(const std::string& filename);
  uint32_t batch_size_;
  uint32_t current_batch = 0;
  nvinfer1::Dims dims_;
  std::vector<std::string> image_filenames_;
};

}  // namespace sss

#endif /* INCLUDE_TRT_CONVERTER_CALIBRATE_STREAM_ */
