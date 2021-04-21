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
    for (const auto& d: dims_string) {
        shape.push_back(StringToValue<int>(d));
    }
    return shape;
}


}  // namespace sss