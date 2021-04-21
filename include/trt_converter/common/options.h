#ifndef INCLUDE_TRT_CONVERTER_COMMON_OPTIONS_
#define INCLUDE_TRT_CONVERTER_COMMON_OPTIONS_

#include <string>
#include <unordered_map>
#include <vector>

#include "NvInfer.h"

namespace sss {

inline constexpr int kDefaultMaxBatch = 1;
inline constexpr int kDefaultWorkspace = 16;
inline constexpr int kDefaultMinTiming = 1;
inline constexpr int kDefaultAvgTiming = 8;

// System default params
inline constexpr int kDefaultDevice = 0;

// Inference default params
inline constexpr int kDefaultBatch = 1;
inline constexpr int kDefaultStreams = 1;
inline constexpr int kDefaultIterations = 10;
inline constexpr int kDefaultWarmUp = 200;
inline constexpr int kDefaultDuration = 3;
inline constexpr int kDefaultSleep = 0;

// Reporting default params
inline constexpr int kDefaultAvgRuns = 10;
inline constexpr float kDefaultPercentile = 99;

using Arguments = std::unordered_map<std::string, std::string>;

using IOFormat = std::pair<nvinfer1::DataType, nvinfer1::TensorFormats>;

using ShapeRange = std::array<std::vector<int>, nvinfer1::EnumMax<nvinfer1::OptProfileSelector>()>;

enum class ModelFormat { kAny = 0, kUff = 1, kCaffe = 2, kONNX = 3 };

struct Options {
  virtual void Parse(Arguments& arguments) = 0;
};

struct UFFInput : Options {
  std::vector<std::pair<std::string, nvinfer1::Dims>> inputs;
  bool is_nhwc = false;
  void Parse(Arguments& arguments) override;
  static void Help(std::ostream& os);
};

struct ModelOptions : public Options {
  ModelFormat format;
  std::string model;
  std::string prototxt;
  std::vector<std::string> outputs;
  void Parse(Arguments& arguments) override;
  static void Help(std::ostream& os);
};

struct BuildOptions : public Options {};

struct InferenceOptions : public Options {
  int batch = kDefaultBatch;
  int iterations = kDefaultIterations;
  int warmup = kDefaultWarmUp;
  int duration = kDefaultDuration;
  int sleep = kDefaultSleep;
  int streams = kDefaultStreams;
  bool overlap = true;
  bool skipTransfers = false;
  bool spin = false;
  bool threads = false;
  bool graph = false;
  bool skip = false;
  bool rerun = false;
  std::unordered_map<std::string, std::string> inputs;
  std::unordered_map<std::string, std::vector<int>> shapes;

  void Parse(Arguments& arguments) override;

  static void Help(std::ostream& out);
};

struct SystemOptions : public Options {};

struct ProfileOptions : public Options {};

}  // namespace sss

#endif /* INCLUDE_TRT_CONVERTER_COMMON_OPTIONS_ */
