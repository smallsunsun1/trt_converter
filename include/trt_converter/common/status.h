#ifndef INCLUDE_TRT_CONVERTER_COMMON_STATUS_
#define INCLUDE_TRT_CONVERTER_COMMON_STATUS_

#include <string>
#include <utility>
#include <variant>

namespace sss {

struct Status {
  Status(int error_code, const char* message) : error_code(error_code), error_message(message) {}
  bool OK() { return error_code == 0; }
  int ErrorCode() { return error_code; }
  const std::string& ErrorMessage() const { return error_message; }
  static Status MakeStatus(int error_code, const char* message) { return Status(error_code, message); }
  int error_code = 0;
  std::string error_message;
};

template <typename T>
class StatusOr {
 public:
  StatusOr(Status status) : data_(status) {}
  template <typename... Args, std::enable_if_t<std::is_convertible_v<T, Args...>, int> = 0>
  StatusOr(Args&&... args) : data_(T(std::forward<Args>(args)...)) {}
  const Status* GetStatusIfPresent() const {
    if (auto val = std::get_if<Status>(&data_)) {
      return val;
    } else {
      return nullptr;
    }
  }
  const T* GetDataIfPresent() const {
    if (auto val = std::get_if<T>(&data_)) {
      return val;
    } else {
      return nullptr;
    }
  }

 private:
  std::variant<Status, T> data_;
};

}  // namespace sss

#endif /* INCLUDE_TRT_CONVERTER_COMMON_STATUS_ */
