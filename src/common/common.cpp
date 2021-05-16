#include "trt_converter/common/common.h"

#include <filesystem>

namespace sss {

namespace fs = std::filesystem;

std::string LocateFile(const std::string& filepath_suffix, const std::vector<std::string>& directories) {
  fs::path suffix(filepath_suffix);
  for (const std::string& directory : directories) {
    fs::path t_directory(directory);
    t_directory /= suffix;
    if (fs::exists(t_directory)) {
      return t_directory.string();
    }
  }
  return {};
}

}  // namespace sss