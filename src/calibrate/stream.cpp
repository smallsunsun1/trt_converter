#include "trt_converter/calibrate/stream.h"

#include <algorithm>
#include <cstring>
#include <fstream>

namespace sss {

std::unique_ptr<float[]> ImageDataStream::GetBatch() {
  const std::string filename = image_filenames_[current_batch];
  std::vector<float> image_data = ReadImageData(filename);
  std::unique_ptr<float[]> final_res = std::make_unique<float[]>(image_data.size());
  memcpy(final_res.get(), image_data.data(), image_data.size() * sizeof(float));
  return final_res;
}

std::vector<float> ImageDataStream::ReadImageData(const std::string& filename) {
  std::ifstream file(filename, std::ios::binary);
  std::istreambuf_iterator<char> begin(file);
  std::istreambuf_iterator<char> end;
  std::string result(begin, end);
  std::vector<float> final_res;
  std::transform(result.begin(), result.end(), std::back_inserter(final_res),
                 [](char value) { return static_cast<float>(value); });
  return final_res;
}

}  // namespace sss