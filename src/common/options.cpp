#include "trt_converter/common/options.h"

#include <regex>

#include "trt_converter/common/common.h"
#include "trt_converter/common/utils.h"
namespace sss {

namespace {

static std::vector<std::string> SplitToStringVec(const std::string& options, const std::string& seperator) {
  std::vector<std::string> result;
  std::regex re(seperator);
  std::sregex_token_iterator begin(options.begin(), options.end(), re, -1);
  std::sregex_token_iterator end;
  for (auto iter = begin; iter != end; ++iter) {
    result.push_back(iter->str());
  }
  return result;
}
template <typename T>
T StringToValue(const std::string& option) {
  return T{option};
}
template <>
int StringToValue<int>(const std::string& option) {
  return std::stoi(option);
}
template <>
float StringToValue<float>(const std::string& option) {
  return std::stof(option);
}
template <>
bool StringToValue<bool>(const std::string&) {
  return true;
}

template <>
std::vector<int> StringToValue<std::vector<int>>(const std::string& option) {
  std::vector<int> shape;
  std::vector<std::string> dims_string = SplitToStringVec(option, "x");
  for (const auto& d : dims_string) {
    shape.push_back(StringToValue<int>(d));
  }
  return shape;
}

template <>
nvinfer1::DataType StringToValue<nvinfer1::DataType>(const std::string& option) {
  const std::unordered_map<std::string, nvinfer1::DataType> str_to_DT{{"fp32", nvinfer1::DataType::kFLOAT},
                                                                      {"fp16", nvinfer1::DataType::kHALF},
                                                                      {"int8", nvinfer1::DataType::kINT8},
                                                                      {"int32", nvinfer1::DataType::kINT32}};
  const auto& dt = str_to_DT.find(option);
  if (dt == str_to_DT.end()) {
    throw std::invalid_argument("Invalid DataType " + option);
  }
  return dt->second;
}

template <>
nvinfer1::TensorFormats StringToValue<nvinfer1::TensorFormats>(const std::string& option) {
  std::vector<std::string> option_strings = SplitToStringVec(option, "+");
  const std::unordered_map<std::string, nvinfer1::TensorFormat> str_to_Fmt{
      {"chw", nvinfer1::TensorFormat::kLINEAR},
      {"chw2", nvinfer1::TensorFormat::kCHW2},
      {"chw4", nvinfer1::TensorFormat::kCHW4},
      {"hwc8", nvinfer1::TensorFormat::kHWC8},
      {"chw16", nvinfer1::TensorFormat::kCHW16},
      {"chw32", nvinfer1::TensorFormat::kCHW32},
      {"dhwc8", nvinfer1::TensorFormat::kDHWC8},
      {"hwc", nvinfer1::TensorFormat::kHWC},
      {"dla_linear", nvinfer1::TensorFormat::kDLA_LINEAR},
      {"dla_hwc4", nvinfer1::TensorFormat::kDLA_HWC4}};
  nvinfer1::TensorFormats formats{};
  for (auto f : option_strings) {
    const auto& tf = str_to_Fmt.find(f);
    if (tf == str_to_Fmt.end()) {
      throw std::invalid_argument(std::string("Invalid TensorFormat ") + f);
    }
    formats |= 1U << int(tf->second);
  }

  return formats;
}

template <>
IOFormat StringToValue<IOFormat>(const std::string& option) {
  IOFormat io_format{};
  const size_t colon = option.find(':');

  if (colon == std::string::npos) {
    throw std::invalid_argument(std::string("Invalid IOFormat ") + option);
  }

  io_format.first = StringToValue<nvinfer1::DataType>(option.substr(0, colon));
  io_format.second = StringToValue<nvinfer1::TensorFormats>(option.substr(colon + 1));

  return io_format;
}

template <typename T>
std::pair<std::string, T> SplitNameAndValue(const std::string& s) {
  std::string tensor_name;
  std::string value_string;
  // Split on the last :
  std::vector<std::string> name_range{SplitToStringVec(s, ":")};
  // Everything before the last : is the name
  tensor_name = name_range[0];
  for (size_t i = 1; i < name_range.size() - 1; i++) {
    tensor_name += ":" + name_range[i];
  }
  // Value is the string element after the last :
  value_string = name_range[name_range.size() - 1];
  return std::pair<std::string, T>(tensor_name, StringToValue<T>(value_string));
}

template <typename T>
void SplitInsertKeyValue(const std::vector<std::string>& kv_list, T& map) {
  for (const auto& kv : kv_list) {
    map.insert(SplitNameAndValue<typename T::mapped_type>(kv));
  }
}

const char* BoolToEnabled(bool enable) { return enable ? "Enabled" : "Disabled"; }

template <typename T>
bool CheckEraseOption(Arguments& arguments, const std::string& option, T& value) {
  auto match = arguments.find(option);
  if (match != arguments.end()) {
    value = StringToValue<T>(match->second);
    arguments.erase(match);
    return true;
  }

  return false;
}

// Like checkEraseOption, but sets value to false if arguments contain the option.
// This function should be used for options that default to true.
bool CheckEraseNegativeOption(Arguments& arguments, const std::string& option, bool& value) {
  bool dummy;
  if (CheckEraseOption(arguments, option, dummy)) {
    value = false;
    return true;
  }
  return false;
}

template <typename T>
bool CheckEraseRepeatedOption(Arguments& arguments, const std::string& option, std::vector<T>& values) {
  auto match = arguments.equal_range(option);
  if (match.first == match.second) {
    return false;
  }
  auto add_value = [&values](Arguments::value_type& value) { values.emplace_back(StringToValue<T>(value.second)); };
  std::for_each(match.first, match.second, add_value);
  arguments.erase(match.first, match.second);
  return true;
}

void InsertShapesBuild(std::unordered_map<std::string, ShapeRange>& shapes, nvinfer1::OptProfileSelector selector,
                       const std::string& name, const std::vector<int>& dims) {
  shapes[name][static_cast<size_t>(selector)] = dims;
}

void InsertShapesInference(std::unordered_map<std::string, std::vector<int>>& shapes, const std::string& name,
                           const std::vector<int>& dims) {
  shapes[name] = dims;
}

std::string RemoveSingleQuotationMarks(std::string& str) {
  std::vector<std::string> str_list{SplitToStringVec(str, "\'")};
  // Remove all the escaped single quotation marks
  std::string retVal = "";
  // Do not really care about unterminated sequences
  for (size_t i = 0; i < str_list.size(); i++) {
    retVal += str_list[i];
  }
  return retVal;
}

bool GetShapesBuild(Arguments& arguments, std::unordered_map<std::string, ShapeRange>& shapes, const char* argument,
                    nvinfer1::OptProfileSelector selector) {
  std::string list;
  bool ret_val = CheckEraseOption(arguments, argument, list);
  std::vector<std::string> shape_list{SplitToStringVec(list, ",")};
  for (const auto& s : shape_list) {
    auto name_dims_pair = SplitNameAndValue<std::vector<int>>(s);
    auto tensor_name = RemoveSingleQuotationMarks(name_dims_pair.first);
    auto dims = name_dims_pair.second;
    InsertShapesBuild(shapes, selector, tensor_name, dims);
  }
  return ret_val;
}

bool GetShapesInference(Arguments& arguments, std::unordered_map<std::string, std::vector<int>>& shapes,
                        const char* argument) {
  std::string list;
  bool ret_val = CheckEraseOption(arguments, argument, list);
  std::vector<std::string> shape_list{SplitToStringVec(list, ",")};
  for (const auto& s : shape_list) {
    auto name_dims_pair = SplitNameAndValue<std::vector<int>>(s);
    auto tensorName = RemoveSingleQuotationMarks(name_dims_pair.first);
    auto dims = name_dims_pair.second;
    InsertShapesInference(shapes, tensorName, dims);
  }
  return ret_val;
}

void ProcessShapes(std::unordered_map<std::string, ShapeRange>& shapes, bool minShapes, bool opt_shapes,
                   bool max_shapes, bool calib) {
  // Only accept optShapes only or all three of minShapes, optShapes, maxShapes
  if (((minShapes || max_shapes) && !opt_shapes)    // minShapes only, maxShapes only, both minShapes and maxShapes
      || (minShapes && !max_shapes && opt_shapes)   // both minShapes and optShapes
      || (!minShapes && max_shapes && opt_shapes))  // both maxShapes and optShapes
  {
    if (calib) {
      throw std::invalid_argument(
          "Must specify only --optShapesCalib or all of --minShapesCalib, --optShapesCalib, --maxShapesCalib");
    } else {
      throw std::invalid_argument("Must specify only --optShapes or all of --minShapes, --optShapes, --maxShapes");
    }
  }

  // If optShapes only, expand optShapes to minShapes and maxShapes
  if (opt_shapes && !minShapes && !max_shapes) {
    std::unordered_map<std::string, ShapeRange> new_shapes;
    for (auto& s : shapes) {
      InsertShapesBuild(new_shapes, nvinfer1::OptProfileSelector::kMIN, s.first,
                        s.second[static_cast<size_t>(nvinfer1::OptProfileSelector::kOPT)]);
      InsertShapesBuild(new_shapes, nvinfer1::OptProfileSelector::kOPT, s.first,
                        s.second[static_cast<size_t>(nvinfer1::OptProfileSelector::kOPT)]);
      InsertShapesBuild(new_shapes, nvinfer1::OptProfileSelector::kMAX, s.first,
                        s.second[static_cast<size_t>(nvinfer1::OptProfileSelector::kOPT)]);
    }
    shapes = new_shapes;
  }
}

template <typename T>
void PrintShapes(std::ostream& os, const char* phase, const T& shapes) {
  if (shapes.empty()) {
    os << "Input " << phase << " shapes: model" << std::endl;
  } else {
    for (const auto& s : shapes) {
      os << "Input " << phase << " shape: " << s.first << "=" << s.second << std::endl;
    }
  }
}

std::ostream& PrintBatch(std::ostream& os, int maxBatch) {
  if (maxBatch) {
    os << maxBatch;
  } else {
    os << "explicit";
  }
  return os;
}

std::ostream& PrintTacticSources(std::ostream& os, nvinfer1::TacticSources enabledSources,
                                 nvinfer1::TacticSources disabledSources) {
  if (!enabledSources && !disabledSources) {
    os << "Using default tactic sources";
  } else {
    uint32_t cublas = 1U << static_cast<uint32_t>(nvinfer1::TacticSource::kCUBLAS);
    uint32_t cublasLt = 1U << static_cast<uint32_t>(nvinfer1::TacticSource::kCUBLAS_LT);

    if (enabledSources & cublas) {
      os << " +cublas";
    }
    if (disabledSources & cublas) {
      os << " -cublas";
    }
    if (enabledSources & cublasLt) {
      os << " +cublasLt";
    }
    if (disabledSources & cublasLt) {
      os << " -cublasLt";
    }
  }
  return os;
}

std::ostream& PrintPrecision(std::ostream& os, const BuildOptions& options) {
  os << "FP32";
  if (options.fp16) {
    os << "+FP16";
  }
  if (options.int8) {
    os << "+INT8";
  }
  return os;
}
}  // namespace

Arguments ArgsToArgumentsMap(int argc, char* argv[]) {
  Arguments arguments;
  for (int i = 1; i < argc; ++i) {
    auto valuePtr = strchr(argv[i], '=');
    if (valuePtr) {
      std::string value{valuePtr + 1};
      arguments.emplace(std::string(argv[i], valuePtr - argv[i]), value);
    } else {
      arguments.emplace(argv[i], "");
    }
  }
  return arguments;
}

void BaseModelOptions::Parse(Arguments& arguments) {
  if (CheckEraseOption(arguments, "--onnx", model)) {
    format = ModelFormat::kONNX;
  } else if (CheckEraseOption(arguments, "--uff", model)) {
    format = ModelFormat::kUff;
  } else if (CheckEraseOption(arguments, "--model", model)) {
    format = ModelFormat::kCaffe;
  }
}

void UFFInput::Parse(Arguments& arguments) {
  CheckEraseOption(arguments, "--uffNHWC", NHWC);
  std::vector<std::string> args;
  if (CheckEraseRepeatedOption(arguments, "--uffInput", args)) {
    for (const auto& i : args) {
      std::vector<std::string> values{SplitToStringVec(i, ",")};
      if (values.size() == 4) {
        nvinfer1::Dims3 dims{std::stoi(values[1]), std::stoi(values[2]), std::stoi(values[3])};
        inputs.emplace_back(values[0], dims);
      } else {
        throw std::invalid_argument(std::string("Invalid uffInput ") + i);
      }
    }
  }
}

void ModelOptions::Parse(Arguments& arguments) {
  base_model.Parse(arguments);

  switch (base_model.format) {
    case ModelFormat::kCaffe: {
      CheckEraseOption(arguments, "--deploy", prototxt);
      break;
    }
    case ModelFormat::kUff: {
      uff_inputs.Parse(arguments);
      if (uff_inputs.inputs.empty()) {
        throw std::invalid_argument("Uff models require at least one input");
      }
      break;
    }
    case ModelFormat::kONNX:
      break;
    case ModelFormat::kAny: {
      if (CheckEraseOption(arguments, "--deploy", prototxt)) {
        base_model.format = ModelFormat::kCaffe;
      }
      break;
    }
  }
  if (base_model.format == ModelFormat::kCaffe || base_model.format == ModelFormat::kUff) {
    std::vector<std::string> outArgs;
    if (CheckEraseRepeatedOption(arguments, "--output", outArgs)) {
      for (const auto& o : outArgs) {
        for (auto& v : SplitToStringVec(o, ",")) {
          outputs.emplace_back(std::move(v));
        }
      }
    }
    if (outputs.empty()) {
      throw std::invalid_argument("Caffe and Uff models require at least one output");
    }
  }
}

void BuildOptions::Parse(Arguments& arguments) {
  auto getFormats = [&arguments](std::vector<IOFormat>& formatsVector, const char* argument) {
    std::string list;
    CheckEraseOption(arguments, argument, list);
    const std::vector<std::string> formats{SplitToStringVec(list, ",")};
    for (const auto& f : formats) {
      formatsVector.push_back(StringToValue<IOFormat>(f));
    }
  };

  getFormats(input_formats, "--inputIOFormats");
  getFormats(output_formats, "--outputIOFormats");

  bool explicitBatch{false};
  CheckEraseOption(arguments, "--explicitBatch", explicitBatch);
  bool minShapes = GetShapesBuild(arguments, shapes, "--minShapes", nvinfer1::OptProfileSelector::kMIN);
  bool optShapes = GetShapesBuild(arguments, shapes, "--optShapes", nvinfer1::OptProfileSelector::kOPT);
  bool maxShapes = GetShapesBuild(arguments, shapes, "--maxShapes", nvinfer1::OptProfileSelector::kMAX);
  ProcessShapes(shapes, minShapes, optShapes, maxShapes, false);
  bool minShapesCalib = GetShapesBuild(arguments, shapes_calib, "--minShapesCalib", nvinfer1::OptProfileSelector::kMIN);
  bool optShapesCalib = GetShapesBuild(arguments, shapes_calib, "--optShapesCalib", nvinfer1::OptProfileSelector::kOPT);
  bool maxShapesCalib = GetShapesBuild(arguments, shapes_calib, "--maxShapesCalib", nvinfer1::OptProfileSelector::kMAX);
  ProcessShapes(shapes_calib, minShapesCalib, optShapesCalib, maxShapesCalib, true);
  explicitBatch = explicitBatch || !shapes.empty();

  int batch{0};
  CheckEraseOption(arguments, "--maxBatch", batch);
  if (explicitBatch && batch) {
    throw std::invalid_argument("Explicit batch or dynamic shapes enabled with implicit maxBatch " +
                                std::to_string(batch));
  }

  if (explicitBatch) {
    max_batch = 0;
  } else {
    if (batch) {
      max_batch = batch;
    }
  }

  CheckEraseOption(arguments, "--workspace", workspace);
  CheckEraseOption(arguments, "--minTiming", min_timing);
  CheckEraseOption(arguments, "--avgTiming", avg_timing);

  bool best{false};
  CheckEraseOption(arguments, "--best", best);
  if (best) {
    int8 = true;
    fp16 = true;
  }

  CheckEraseOption(arguments, "--refit", refittable);
  CheckEraseNegativeOption(arguments, "--noTF32", tf32);
  CheckEraseOption(arguments, "--fp16", fp16);
  CheckEraseOption(arguments, "--int8", int8);
  CheckEraseOption(arguments, "--safe", safe);
  bool calibCheck = CheckEraseOption(arguments, "--calib", calibration);
  if (int8 && calibCheck && !shapes.empty() && shapes_calib.empty()) {
    shapes_calib = shapes;
  }
  CheckEraseNegativeOption(arguments, "--noBuilderCache", builder_cache);

  std::string nvtxModeString;
  CheckEraseOption(arguments, "--nvtxMode", nvtxModeString);
  if (nvtxModeString == "default") {
    nvtx_mode = nvinfer1::ProfilingVerbosity::kDEFAULT;
  } else if (nvtxModeString == "none") {
    nvtx_mode = nvinfer1::ProfilingVerbosity::kNONE;
  } else if (nvtxModeString == "verbose") {
    nvtx_mode = nvinfer1::ProfilingVerbosity::kVERBOSE;
  } else if (!nvtxModeString.empty()) {
    throw std::invalid_argument(std::string("Unknown nvtxMode: ") + nvtxModeString);
  }

  if (CheckEraseOption(arguments, "--loadEngine", engine)) {
    load = true;
  }
  if (CheckEraseOption(arguments, "--saveEngine", engine)) {
    save = true;
  }
  if (load && save) {
    throw std::invalid_argument("Incompatible load and save engine options selected");
  }

  std::string tacticSourceArgs;
  if (CheckEraseOption(arguments, "--tacticSources", tacticSourceArgs)) {
    std::vector<std::string> tacticList = SplitToStringVec(tacticSourceArgs, ",");
    for (auto& t : tacticList) {
      bool enable{false};
      if (t.front() == '+') {
        enable = true;
      } else if (t.front() != '-') {
        throw std::invalid_argument("Tactic conditional (+|-) is missing");
      }
      t.erase(0, 1);

      nvinfer1::TacticSource source{};
      if (t == "cublas") {
        source = nvinfer1::TacticSource::kCUBLAS;
      } else if (t == "cublasLt") {
        source = nvinfer1::TacticSource::kCUBLAS_LT;
      } else {
        throw std::invalid_argument(std::string("Unknown tactic source: ") + t);
      }

      uint32_t sourceBit = 1U << static_cast<uint32_t>(source);

      if (enable) {
        enabled_tactics |= sourceBit;
      } else {
        disabled_tactics |= sourceBit;
      }

      if (enabled_tactics & disabled_tactics) {
        throw std::invalid_argument(std::string("Cannot enable and disable ") + t);
      }
    }
  }
}

void SystemOptions::Parse(Arguments& arguments) {
  CheckEraseOption(arguments, "--device", device);
  CheckEraseOption(arguments, "--useDLACore", DLACore);
  CheckEraseOption(arguments, "--allowGPUFallback", fallback);
  std::string pluginName;
  while (CheckEraseOption(arguments, "--plugins", pluginName)) {
    plugins.emplace_back(pluginName);
  }
}

void InferenceOptions::Parse(Arguments& arguments) {
  CheckEraseOption(arguments, "--streams", streams);
  CheckEraseOption(arguments, "--iterations", iterations);
  CheckEraseOption(arguments, "--duration", duration);
  CheckEraseOption(arguments, "--warmUp", warmup);
  CheckEraseOption(arguments, "--sleepTime", sleep);
  bool exposeDMA{false};
  if (CheckEraseOption(arguments, "--exposeDMA", exposeDMA)) {
    overlap = !exposeDMA;
  }
  CheckEraseOption(arguments, "--noDataTransfers", skip_transfers);
  CheckEraseOption(arguments, "--useSpinWait", spin);
  CheckEraseOption(arguments, "--threads", threads);
  CheckEraseOption(arguments, "--useCudaGraph", graph);
  CheckEraseOption(arguments, "--separateProfileRun", rerun);
  CheckEraseOption(arguments, "--buildOnly", skip);

  std::string list;
  CheckEraseOption(arguments, "--loadInputs", list);
  std::vector<std::string> inputsList{SplitToStringVec(list, ",")};
  SplitInsertKeyValue(inputsList, inputs);

  GetShapesInference(arguments, shapes, "--shapes");

  int batchOpt{0};
  CheckEraseOption(arguments, "--batch", batchOpt);
  if (!shapes.empty() && batchOpt) {
    throw std::invalid_argument("Explicit batch or dynamic shapes enabled with implicit batch " +
                                std::to_string(batchOpt));
  }
  if (batchOpt) {
    batch = batchOpt;
  } else {
    if (!shapes.empty()) {
      batch = 0;
    }
  }
}

void ReportingOptions::Parse(Arguments& arguments) {
  CheckEraseOption(arguments, "--percentile", percentile);
  CheckEraseOption(arguments, "--avgRuns", avgs);
  CheckEraseOption(arguments, "--verbose", verbose);
  CheckEraseOption(arguments, "--dumpRefit", refit);
  CheckEraseOption(arguments, "--dumpOutput", output);
  CheckEraseOption(arguments, "--dumpProfile", profile);
  CheckEraseOption(arguments, "--exportTimes", export_times);
  CheckEraseOption(arguments, "--exportOutput", export_output);
  CheckEraseOption(arguments, "--exportProfile", export_profile);
  if (percentile < 0 || percentile > 100) {
    throw std::invalid_argument(std::string("Percentile ") + std::to_string(percentile) + "is not in [0,100]");
  }
}

bool parseHelp(Arguments& arguments) {
  bool helpLong{false};
  bool helpShort{false};
  CheckEraseOption(arguments, "--help", helpLong);
  CheckEraseOption(arguments, "-h", helpShort);
  return helpLong || helpShort;
}

void AllOptions::Parse(Arguments& arguments) {
  model.Parse(arguments);
  build.Parse(arguments);
  system.Parse(arguments);
  inference.Parse(arguments);

  if (model.base_model.format == ModelFormat::kONNX) {
    build.max_batch = 0;  // ONNX only supports explicit batch mode.
  }

  if ((!build.max_batch && inference.batch && inference.batch != kDefaultBatch && !build.shapes.empty()) ||
      (build.max_batch && build.max_batch != kDefaultMaxBatch && !inference.batch)) {
    // If either has selected implict batch and the other has selected explicit batch
    throw std::invalid_argument("Conflicting build and inference batch settings");
  }

  if (build.shapes.empty() && !inference.shapes.empty()) {
    for (auto& s : inference.shapes) {
      InsertShapesBuild(build.shapes, nvinfer1::OptProfileSelector::kMIN, s.first, s.second);
      InsertShapesBuild(build.shapes, nvinfer1::OptProfileSelector::kOPT, s.first, s.second);
      InsertShapesBuild(build.shapes, nvinfer1::OptProfileSelector::kMAX, s.first, s.second);
    }
    build.max_batch = 0;
  } else {
    if (!build.shapes.empty() && inference.shapes.empty()) {
      for (auto& s : build.shapes) {
        InsertShapesInference(inference.shapes, s.first,
                              s.second[static_cast<size_t>(nvinfer1::OptProfileSelector::kOPT)]);
      }
    }
    if (!build.max_batch) {
      inference.batch = 0;
    }
  }

  if (build.max_batch && inference.batch) {
    // For implicit batch, check for compatibility and if --maxBatch is not given and inference batch is greater
    // than maxBatch, use inference batch also for maxBatch
    if (build.max_batch != kDefaultMaxBatch && build.max_batch < inference.batch) {
      throw std::invalid_argument("Build max batch " + std::to_string(build.max_batch) +
                                  " is less than inference batch " + std::to_string(inference.batch));
    } else {
      if (build.max_batch < inference.batch) {
        build.max_batch = inference.batch;
      }
    }
  }

  reporting.Parse(arguments);
  helps = parseHelp(arguments);

  if (!helps) {
    if (!build.load && model.base_model.format == ModelFormat::kAny) {
      throw std::invalid_argument("Model missing or format not recognized");
    }
    if (!build.load && !build.max_batch && model.base_model.format != ModelFormat::kONNX) {
      throw std::invalid_argument("Explicit batch size not supported for Caffe and Uff models");
    }
    if (build.safe && system.DLACore >= 0) {
      auto checkSafeDLAFormats = [](const std::vector<IOFormat>& fmt) {
        return fmt.empty() ? false : std::all_of(fmt.begin(), fmt.end(), [](const IOFormat& pair) {
          bool supported{false};
          const bool isCHW4{pair.second == 1U << static_cast<int>(nvinfer1::TensorFormat::kCHW4)};
          const bool isCHW32{pair.second == 1U << static_cast<int>(nvinfer1::TensorFormat::kCHW32)};
          const bool isCHW16{pair.second == 1U << static_cast<int>(nvinfer1::TensorFormat::kCHW16)};
          supported |= pair.first == nvinfer1::DataType::kINT8 && (isCHW4 || isCHW32);
          supported |= pair.first == nvinfer1::DataType::kHALF && (isCHW4 || isCHW16);
          return supported;
        });
      };
      if (!checkSafeDLAFormats(build.input_formats) || !checkSafeDLAFormats(build.input_formats)) {
        throw std::invalid_argument("I/O formats for safe DLA capability are restricted to fp16:chw16 or int8:chw32");
      }
      if (system.fallback) {
        throw std::invalid_argument("GPU fallback (--allowGPUFallback) not allowed for safe DLA capability");
      }
    }
  }
}

std::ostream& operator<<(std::ostream& os, const BaseModelOptions& options) {
  os << "=== Model Options ===" << std::endl;

  os << "Format: ";
  switch (options.format) {
    case ModelFormat::kCaffe: {
      os << "Caffe";
      break;
    }
    case ModelFormat::kONNX: {
      os << "ONNX";
      break;
    }
    case ModelFormat::kUff: {
      os << "UFF";
      break;
    }
    case ModelFormat::kAny:
      os << "*";
      break;
  }
  os << std::endl << "Model: " << options.model << std::endl;

  return os;
}

std::ostream& operator<<(std::ostream& os, const UFFInput& input) {
  os << "Uff Inputs Layout: " << (input.NHWC ? "NHWC" : "NCHW") << std::endl;
  for (const auto& i : input.inputs) {
    os << "Input: " << i.first << "," << i.second.d[0] << "," << i.second.d[1] << "," << i.second.d[2] << std::endl;
  }

  return os;
}

std::ostream& operator<<(std::ostream& os, const ModelOptions& options) {
  os << options.base_model;
  switch (options.base_model.format) {
    case ModelFormat::kCaffe: {
      os << "Prototxt: " << options.prototxt << std::endl;
      break;
    }
    case ModelFormat::kUff: {
      os << options.uff_inputs;
      break;
    }
    case ModelFormat::kONNX:  // Fallthrough: No options to report for ONNX or the generic case
    case ModelFormat::kAny:
      break;
  }

  os << "Output:";
  for (const auto& o : options.outputs) {
    os << " " << o;
  }
  os << std::endl;

  return os;
}

std::ostream& operator<<(std::ostream& os, const IOFormat& format) {
  switch (format.first) {
    case nvinfer1::DataType::kFLOAT: {
      os << "fp32:";
      break;
    }
    case nvinfer1::DataType::kHALF: {
      os << "fp16:";
      break;
    }
    case nvinfer1::DataType::kINT8: {
      os << "int8:";
      break;
    }
    case nvinfer1::DataType::kINT32: {
      os << "int32:";
      break;
    }
    case nvinfer1::DataType::kBOOL: {
      os << "Bool:";
      break;
    }
  }

  for (int f = 0; f < nvinfer1::EnumMax<nvinfer1::TensorFormat>(); ++f) {
    if ((1U << f) & format.second) {
      if (f) {
        os << "+";
      }
      switch (nvinfer1::TensorFormat(f)) {
        case nvinfer1::TensorFormat::kLINEAR: {
          os << "chw";
          break;
        }
        case nvinfer1::TensorFormat::kCHW2: {
          os << "chw2";
          break;
        }
        case nvinfer1::TensorFormat::kHWC8: {
          os << "hwc8";
          break;
        }
        case nvinfer1::TensorFormat::kCHW4: {
          os << "chw4";
          break;
        }
        case nvinfer1::TensorFormat::kCHW16: {
          os << "chw16";
          break;
        }
        case nvinfer1::TensorFormat::kCHW32: {
          os << "chw32";
          break;
        }
        case nvinfer1::TensorFormat::kDHWC8: {
          os << "dhwc8";
          break;
        }
        case nvinfer1::TensorFormat::kCDHW32: {
          os << "cdhw32";
          break;
        }
        case nvinfer1::TensorFormat::kHWC: {
          os << "hwc";
          break;
        }
        case nvinfer1::TensorFormat::kDLA_LINEAR: {
          os << "dla_linear";
          break;
        }
        case nvinfer1::TensorFormat::kDLA_HWC4: {
          os << "dla_hwc4";
          break;
        }
      }
    }
  }
  return os;
};

std::ostream& operator<<(std::ostream& os, const ShapeRange& dims) {
  int i = 0;
  for (const auto& d : dims) {
    if (!d.size()) {
      break;
    }
    os << (i ? "+" : "") << d;
    ++i;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const BuildOptions& options) {
  // clang-format off
    os << "=== Build Options ==="                                                                                       << std::endl <<

          "Max batch: ";        PrintBatch(os, options.max_batch)                                                        << std::endl <<
          "Workspace: "      << options.workspace << " MiB"                                                             << std::endl <<
          "minTiming: "      << options.min_timing                                                                       << std::endl <<
          "avgTiming: "      << options.avg_timing                                                                       << std::endl <<
          "Precision: ";        PrintPrecision(os, options)                                                             << std::endl <<
          "Calibration: "    << (options.int8 && options.calibration.empty() ? "Dynamic" : options.calibration.c_str()) << std::endl <<
          "Refit: "          << BoolToEnabled(options.refittable)                                                       << std::endl <<
          "Safe mode: "      << BoolToEnabled(options.safe)                                                             << std::endl <<
          "Save engine: "    << (options.save ? options.engine : "")                                                    << std::endl <<
          "Load engine: "    << (options.load ? options.engine : "")                                                    << std::endl <<
          "Builder Cache: "  << BoolToEnabled(options.builder_cache)                                                     << std::endl <<
          "NVTX verbosity: " << static_cast<int>(options.nvtx_mode)                                                      << std::endl <<
          "Tactic sources: ";   PrintTacticSources(os, options.enabled_tactics, options.disabled_tactics)                 << std::endl;
  // clang-format on

  auto printIOFormats = [](std::ostream& os, const char* direction, const std::vector<IOFormat> formats) {
    if (formats.empty()) {
      os << direction << "s format: fp32:CHW" << std::endl;
    } else {
      for (const auto& f : formats) {
        os << direction << ": " << f << std::endl;
      }
    }
  };

  printIOFormats(os, "Input(s)", options.input_formats);
  printIOFormats(os, "Output(s)", options.output_formats);
  PrintShapes(os, "build", options.shapes);
  PrintShapes(os, "calibration", options.shapes_calib);

  return os;
}

std::ostream& operator<<(std::ostream& os, const SystemOptions& options) {
  // clang-format off
    os << "=== System Options ==="                                                                << std::endl <<

          "Device: "  << options.device                                                           << std::endl <<
          "DLACore: " << (options.DLACore != -1 ? std::to_string(options.DLACore) : "")           <<
                         (options.DLACore != -1 && options.fallback ? "(With GPU fallback)" : "") << std::endl;
  // clang-format on
  os << "Plugins:";

  for (const auto& p : options.plugins) {
    os << " " << p;
  }
  os << std::endl;

  return os;
}

std::ostream& operator<<(std::ostream& os, const InferenceOptions& options) {
  // clang-format off
    os << "=== Inference Options ==="                               << std::endl <<

          "Batch: ";
    if (options.batch && options.shapes.empty())
    {
                          os << options.batch                       << std::endl;
    }
    else
    {
                          os << "Explicit"                          << std::endl;
    }
    PrintShapes(os, "inference", options.shapes);
    os << "Iterations: "          << options.iterations                    << std::endl <<
          "Duration: "            << options.duration   << "s (+ "
                                  << options.warmup     << "ms warm up)"   << std::endl <<
          "Sleep time: "          << options.sleep      << "ms"            << std::endl <<
          "Streams: "             << options.streams                       << std::endl <<
          "ExposeDMA: "           << BoolToEnabled(!options.overlap)       << std::endl <<
          "Data transfers: "      << BoolToEnabled(!options.skip_transfers) << std::endl <<
          "Spin-wait: "           << BoolToEnabled(options.spin)           << std::endl <<
          "Multithreading: "      << BoolToEnabled(options.threads)        << std::endl <<
          "CUDA Graph: "          << BoolToEnabled(options.graph)          << std::endl <<
          "Separate profiling: "  << BoolToEnabled(options.rerun)          << std::endl <<
          "Skip inference: "      << BoolToEnabled(options.skip)           << std::endl;

  // clang-format on
  os << "Inputs:" << std::endl;
  for (const auto& input : options.inputs) {
    os << input.first << "<-" << input.second << std::endl;
  }

  return os;
}

std::ostream& operator<<(std::ostream& os, const ReportingOptions& options) {
  // clang-format off
    os << "=== Reporting Options ==="                                               << std::endl <<

          "Verbose: "                          << BoolToEnabled(options.verbose)    << std::endl <<
          "Averages: "                         << options.avgs << " inferences"     << std::endl <<
          "Percentile: "                       << options.percentile                << std::endl <<
          "Dump refittable layers:"            << BoolToEnabled(options.refit)      << std::endl <<
          "Dump output: "                      << BoolToEnabled(options.output)     << std::endl <<
          "Profile: "                          << BoolToEnabled(options.profile)    << std::endl <<
          "Export timing to JSON file: "       << options.export_times               << std::endl <<
          "Export output to JSON file: "       << options.export_output              << std::endl <<
          "Export profile to JSON file: "      << options.export_profile             << std::endl;
  // clang-format on

  return os;
}

std::ostream& operator<<(std::ostream& os, const AllOptions& options) {
  os << options.model << options.build << options.system << options.inference << options.reporting << std::endl;
  return os;
}

void BaseModelOptions::help(std::ostream& os) {
  // clang-format off
    os << "  --uff=<file>                UFF model"                                             << std::endl <<
          "  --onnx=<file>               ONNX model"                                            << std::endl <<
          "  --model=<file>              Caffe model (default = no model, random weights used)" << std::endl;
  // clang-format on
}

void UFFInput::help(std::ostream& os) {
  // clang-format off
    os << "  --uffInput=<name>,X,Y,Z     Input blob name and its dimensions (X,Y,Z=C,H,W), it can be specified "
                                                       "multiple times; at least one is required for UFF models" << std::endl <<
          "  --uffNHWC                   Set if inputs are in the NHWC layout instead of NCHW (use "             <<
                                                                    "X,Y,Z=H,W,C order in --uffInput)"           << std::endl;
  // clang-format on
}

void ModelOptions::help(std::ostream& os) {
  // clang-format off
    os << "=== Model Options ==="                                                                                 << std::endl;
    BaseModelOptions::help(os);
    os << "  --deploy=<file>             Caffe prototxt file"                                                     << std::endl <<
          "  --output=<name>[,<name>]*   Output names (it can be specified multiple times); at least one output "
                                                                                  "is required for UFF and Caffe" << std::endl;
    UFFInput::help(os);
  // clang-format on
}

void BuildOptions::help(std::ostream& os) {
  // clang-format off
    os << "=== Build Options ==="                                                                                                            << std::endl <<

          "  --maxBatch                  Set max batch size and build an implicit batch engine (default = " << kDefaultMaxBatch << ")"        << std::endl <<
          "  --explicitBatch             Use explicit batch sizes when building the engine (default = implicit)"                             << std::endl <<
          "  --minShapes=spec            Build with dynamic shapes using a profile with the min shapes provided"                             << std::endl <<
          "  --optShapes=spec            Build with dynamic shapes using a profile with the opt shapes provided"                             << std::endl <<
          "  --maxShapes=spec            Build with dynamic shapes using a profile with the max shapes provided"                             << std::endl <<
          "  --minShapesCalib=spec       Calibrate with dynamic shapes using a profile with the min shapes provided"                         << std::endl <<
          "  --optShapesCalib=spec       Calibrate with dynamic shapes using a profile with the opt shapes provided"                         << std::endl <<
          "  --maxShapesCalib=spec       Calibrate with dynamic shapes using a profile with the max shapes provided"                         << std::endl <<
          "                              Note: All three of min, opt and max shapes must be supplied."                                       << std::endl <<
          "                                    However, if only opt shapes is supplied then it will be expanded so"                          << std::endl <<
          "                                    that min shapes and max shapes are set to the same values as opt shapes."                     << std::endl <<
          "                                    In addition, use of dynamic shapes implies explicit batch."                                   << std::endl <<
          "                                    Input names can be wrapped with escaped single quotes (ex: \\\'Input:0\\\')."                 << std::endl <<
          "                              Example input shapes spec: input0:1x3x256x256,input1:1x3x128x128"                                   << std::endl <<
          "                              Each input shape is supplied as a key-value pair where key is the input name and"                   << std::endl <<
          "                              value is the dimensions (including the batch dimension) to be used for that input."                 << std::endl <<
          "                              Each key-value pair has the key and value separated using a colon (:)."                             << std::endl <<
          "                              Multiple input shapes can be provided via comma-separated key-value pairs."                         << std::endl <<
          "  --inputIOFormats=spec       Type and format of each of the input tensors (default = all inputs in fp32:chw)"                    << std::endl <<
          "                              See --outputIOFormats help for the grammar of type and format list."                                << std::endl <<
          "                              Note: If this option is specified, please set comma-separated types and formats for all"            << std::endl <<
          "                                    inputs following the same order as network inputs ID (even if only one input"                 << std::endl <<
          "                                    needs specifying IO format) or set the type and format once for broadcasting."                << std::endl <<
          "  --outputIOFormats=spec      Type and format of each of the output tensors (default = all outputs in fp32:chw)"                  << std::endl <<
          "                              Note: If this option is specified, please set comma-separated types and formats for all"            << std::endl <<
          "                                    outputs following the same order as network outputs ID (even if only one output"              << std::endl <<
          "                                    needs specifying IO format) or set the type and format once for broadcasting."                << std::endl <<
          "                              IO Formats: spec  ::= IOfmt[\",\"spec]"                                                             << std::endl <<
          "                                          IOfmt ::= type:fmt"                                                                     << std::endl <<
          "                                          type  ::= \"fp32\"|\"fp16\"|\"int32\"|\"int8\""                                         << std::endl <<
          "                                          fmt   ::= (\"chw\"|\"chw2\"|\"chw4\"|\"hwc8\"|\"chw16\"|\"chw32\"|\"dhwc8\")[\"+\"fmt]" << std::endl <<
          "  --workspace=N               Set workspace size in megabytes (default = "                      << kDefaultWorkspace << ")"        << std::endl <<
          "  --noBuilderCache            Disable timing cache in builder (default is to enable timing cache)"                                << std::endl <<
          "  --nvtxMode=mode             Specify NVTX annotation verbosity. mode ::= default|verbose|none"                                   << std::endl <<
          "  --minTiming=M               Set the minimum number of iterations used in kernel selection (default = "
                                                                                                           << kDefaultMinTiming << ")"        << std::endl <<
          "  --avgTiming=M               Set the number of times averaged in each iteration for kernel selection (default = "
                                                                                                           << kDefaultAvgTiming << ")"        << std::endl <<
          "  --noTF32                    Disable tf32 precision (default is to enable tf32, in addition to fp32)"                            << std::endl <<
          "  --refit                     Mark the engine as refittable. This will allow the inspection of refittable layers "                << std::endl <<
          "                              and weights within the engine."                                                                     << std::endl <<
          "  --fp16                      Enable fp16 precision, in addition to fp32 (default = disabled)"                                    << std::endl <<
          "  --int8                      Enable int8 precision, in addition to fp32 (default = disabled)"                                    << std::endl <<
          "  --best                      Enable all precisions to achieve the best performance (default = disabled)"                         << std::endl <<
          "  --calib=<file>              Read INT8 calibration cache file"                                                                   << std::endl <<
          "  --safe                      Only test the functionality available in safety restricted flows"                                   << std::endl <<
          "  --saveEngine=<file>         Save the serialized engine"                                                                         << std::endl <<
          "  --loadEngine=<file>         Load a serialized engine"                                                                           << std::endl <<
          "  --tacticSources=tactics     Specify the tactics to be used by adding (+) or removing (-) tactics from the default "             << std::endl <<
          "                              tactic sources (default = all available tactics)."                                                  << std::endl <<
          "                              Note: Currently only cuBLAS and cuBLAS LT are listed as optional tactics."                          << std::endl <<
          "                              Tactic Sources: tactics ::= [\",\"tactic]"                                                          << std::endl <<
          "                                              tactic  ::= (+|-)lib"                                                               << std::endl <<
          "                                              lib     ::= \"cublas\"|\"cublasLt\""                                                << std::endl;
  // clang-format on
}

void SystemOptions::help(std::ostream& os) {
  // clang-format off
    os << "=== System Options ==="                                                                         << std::endl <<
          "  --device=N                  Select cuda device N (default = "         << kDefaultDevice << ")" << std::endl <<
          "  --useDLACore=N              Select DLA core N for layers that support DLA (default = none)"   << std::endl <<
          "  --allowGPUFallback          When DLA is enabled, allow GPU fallback for unsupported layers "
                                                                                    "(default = disabled)" << std::endl;
    os << "  --plugins                   Plugin library (.so) to load (can be specified multiple times)"   << std::endl;
  // clang-format on
}

void InferenceOptions::help(std::ostream& os) {
  // clang-format off
    os << "=== Inference Options ==="                                                                                               << std::endl <<
          "  --batch=N                   Set batch size for implicit batch engines (default = "              << kDefaultBatch << ")" << std::endl <<
          "  --shapes=spec               Set input shapes for dynamic shapes inference inputs."                                     << std::endl <<
          "                              Note: Use of dynamic shapes implies explicit batch."                                       << std::endl <<
          "                                    Input names can be wrapped with escaped single quotes (ex: \\\'Input:0\\\')."        << std::endl <<
          "                              Example input shapes spec: input0:1x3x256x256, input1:1x3x128x128"                         << std::endl <<
          "                              Each input shape is supplied as a key-value pair where key is the input name and"          << std::endl <<
          "                              value is the dimensions (including the batch dimension) to be used for that input."        << std::endl <<
          "                              Each key-value pair has the key and value separated using a colon (:)."                    << std::endl <<
          "                              Multiple input shapes can be provided via comma-separated key-value pairs."                << std::endl <<
          "  --loadInputs=spec           Load input values from files (default = generate random inputs). Input names can be "
                                                                                       "wrapped with single quotes (ex: 'Input:0')" << std::endl <<
          "                              Input values spec ::= Ival[\",\"spec]"                                                     << std::endl <<
          "                                           Ival ::= name\":\"file"                                                       << std::endl <<
          "  --iterations=N              Run at least N inference iterations (default = "               << kDefaultIterations << ")" << std::endl <<
          "  --warmUp=N                  Run for N milliseconds to warmup before measuring performance (default = "
                                                                                                            << kDefaultWarmUp << ")" << std::endl <<
          "  --duration=N                Run performance measurements for at least N seconds wallclock time (default = "
                                                                                                          << kDefaultDuration << ")" << std::endl <<
          "  --sleepTime=N               Delay inference start with a gap of N milliseconds between launch and compute "
                                                                                               "(default = " << kDefaultSleep << ")" << std::endl <<
          "  --streams=N                 Instantiate N engines to use concurrently (default = "            << kDefaultStreams << ")" << std::endl <<
          "  --exposeDMA                 Serialize DMA transfers to and from device. (default = disabled)"                          << std::endl <<
          "  --noDataTransfers           Do not transfer data to and from the device during inference. (default = disabled)"        << std::endl <<
          "  --useSpinWait               Actively synchronize on GPU events. This option may decrease synchronization time but "
                                                                             "increase CPU usage and power (default = disabled)"    << std::endl <<
          "  --threads                   Enable multithreading to drive engines with independent threads (default = disabled)"      << std::endl <<
          "  --useCudaGraph              Use cuda graph to capture engine execution and then launch inference (default = disabled)" << std::endl <<
          "  --separateProfileRun        Do not attach the profiler in the benchmark run; if profiling is enabled, a second "
                                                                                "profile run will be executed (default = disabled)" << std::endl <<
          "  --buildOnly                 Skip inference perf measurement (default = disabled)"                                      << std::endl;
  // clang-format on
}

void ReportingOptions::help(std::ostream& os) {
  // clang-format off
    os << "=== Reporting Options ==="                                                                    << std::endl <<
          "  --verbose                   Use verbose logging (default = false)"                          << std::endl <<
          "  --avgRuns=N                 Report performance measurements averaged over N consecutive "
                                                       "iterations (default = " << kDefaultAvgRuns << ")" << std::endl <<
          "  --percentile=P              Report performance for the P percentage (0<=P<=100, 0 "
                                        "representing max perf, and 100 representing min perf; (default"
                                                                      " = " << kDefaultPercentile << "%)" << std::endl <<
          "  --dumpRefit                 Print the refittable layers and weights from a refittable "
                                        "engine"                                                         << std::endl <<
          "  --dumpOutput                Print the output tensor(s) of the last inference iteration "
                                                                                  "(default = disabled)" << std::endl <<
          "  --dumpProfile               Print profile information per layer (default = disabled)"       << std::endl <<
          "  --exportTimes=<file>        Write the timing results in a json file (default = disabled)"   << std::endl <<
          "  --exportOutput=<file>       Write the output tensors to a json file (default = disabled)"   << std::endl <<
          "  --exportProfile=<file>      Write the profile information per layer in a json file "
                                                                              "(default = disabled)"     << std::endl;
  // clang-format on
}

void helpHelp(std::ostream& os) {
  // clang-format off
    os << "=== Help ==="                                     << std::endl <<
          "  --help, -h                  Print this message" << std::endl;
  // clang-format on
}

void AllOptions::help(std::ostream& os) {
  ModelOptions::help(os);
  os << std::endl;
  BuildOptions::help(os);
  os << std::endl;
  InferenceOptions::help(os);
  os << std::endl;
  // clang-format off
    os << "=== Build and Inference Batch Options ==="                                                                   << std::endl <<
          "                              When using implicit batch, the max batch size of the engine, if not given, "   << std::endl <<
          "                              is set to the inference batch size;"                                           << std::endl <<
          "                              when using explicit batch, if shapes are specified only for inference, they "  << std::endl <<
          "                              will be used also as min/opt/max in the build profile; if shapes are "         << std::endl <<
          "                              specified only for the build, the opt shapes will be used also for inference;" << std::endl <<
          "                              if both are specified, they must be compatible; and if explicit batch is "     << std::endl <<
          "                              enabled but neither is specified, the model must provide complete static"      << std::endl <<
          "                              dimensions, including batch size, for all inputs"                              << std::endl <<
    std::endl;
  // clang-format on
  ReportingOptions::help(os);
  os << std::endl;
  SystemOptions::help(os);
  os << std::endl;
  helpHelp(os);
}

}  // namespace sss