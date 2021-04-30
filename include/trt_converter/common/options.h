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

struct InferenceOptions : public Options {
  int batch = kDefaultBatch;
  int iterations = kDefaultIterations;
  int warmup = kDefaultWarmUp;
  int duration = kDefaultDuration;
  int sleep = kDefaultSleep;
  int streams = kDefaultStreams;
  bool overlap = true;
  bool skip_transfers = false;
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

struct BuildOptions : public Options {
  int max_batch = kDefaultBatch;  // Parsing sets maxBatch to 0 if explicitBatch is true
  int workspace = kDefaultWorkspace;
  int min_timing = kDefaultMinTiming;
  int avg_timing = kDefaultAvgRuns;
  bool tf32 = true;
  bool refittable = false;
  bool fp16 = false;
  bool int8 = false;
  bool safe = false;
  bool save = false;
  bool load = false;
  bool builder_cache = true;
  nvinfer1::ProfilingVerbosity nvtx_mode{nvinfer1::ProfilingVerbosity::kDEFAULT};
  std::string engine;
  std::string calibration;
  std::unordered_map<std::string, ShapeRange> shapes;
  std::unordered_map<std::string, ShapeRange> shapes_calib;
  std::vector<IOFormat> input_formats;
  std::vector<IOFormat> output_formats;
  nvinfer1::TacticSources enabled_tactics{0};
  nvinfer1::TacticSources disabled_tactics{0};
  void Parse(Arguments& arguments) override;

  static void help(std::ostream& out);
};

struct SystemOptions : public Options {
  int device = kDefaultDevice;
  int DLACore = -1;
  bool fallback{false};
  std::vector<std::string> plugins;

  void Parse(Arguments& arguments) override;

  static void help(std::ostream& out);
};

struct ReportingOptions : public Options {
  bool verbose = false;
  int avgs = kDefaultAvgRuns;
  float percentile = kDefaultPercentile;
  bool refit = false;
  bool output = false;
  bool profile = false;
  std::string exportTimes;
  std::string exportOutput;
  std::string exportProfile;

  void Parse(Arguments& arguments) override;

  static void help(std::ostream& out);
};

struct AllOptions : public Options {
  ModelOptions model;
  BuildOptions build;
  SystemOptions system;
  InferenceOptions inference;
  ReportingOptions reporting;
  bool helps{false};

  void Parse(Arguments& arguments) override;

  static void help(std::ostream& out);
};

}  // namespace sss

#endif /* INCLUDE_TRT_CONVERTER_COMMON_OPTIONS_ */
