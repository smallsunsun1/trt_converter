#ifndef INCLUDE_TRT_CONVERTER_COMMON_COMMON_
#define INCLUDE_TRT_CONVERTER_COMMON_COMMON_

#include <iostream>
#include <vector>

#include "NvInfer.h"

namespace sss {

inline std::ostream& operator<<(std::ostream& os, const nvinfer1::Dims& dims) {
  os << "(";
  for (int i = 0; i < dims.nbDims; ++i) {
    os << (i ? ", " : "") << dims.d[i];
  }
  return os << ")";
}

inline std::ostream& operator<<(std::ostream& os, const std::vector<int>& vec) {
  for (int i = 0, e = static_cast<int>(vec.size()); i < e; ++i) {
    os << (i ? "x" : "") << vec[i];
  }
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const nvinfer1::WeightsRole role) {
  switch (role) {
    case nvinfer1::WeightsRole::kKERNEL: {
      os << "Kernel";
      break;
    }
    case nvinfer1::WeightsRole::kBIAS: {
      os << "Bias";
      break;
    }
    case nvinfer1::WeightsRole::kSHIFT: {
      os << "Shift";
      break;
    }
    case nvinfer1::WeightsRole::kSCALE: {
      os << "Scale";
      break;
    }
    case nvinfer1::WeightsRole::kCONSTANT: {
      os << "Constant";
      break;
    }
  }

  return os;
}

inline unsigned int GetElementSize(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kINT8:
      return 1;
  }
  throw std::runtime_error("Invalid DataType.");
  return 0;
}

}  // namespace sss

#endif /* INCLUDE_TRT_CONVERTER_COMMON_COMMON_ */
