#include "gtest/gtest.h"
#include <filesystem>

#include "trt_converter/common/common.h"

TEST(FILESYSTEM, PATH) {
  std::string suffix = "sss.prototxt";
  std::vector<std::string> directories = {"unknown_path", "./data"};
  std::string result = sss::LocateFile(suffix, directories);
  std::string expected = "./data/sss.prototxt";
  EXPECT_EQ(result, expected);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
