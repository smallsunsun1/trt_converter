#include "trt_converter/common/options.h"

#include <regex>
namespace sss {

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
int StringToValue(const std::string& option) {
  return std::stoi(option);
}

}  // namespace sss