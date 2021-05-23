#include "gtest/gtest.h"
#include <regex>
#include <string>
#include <vector>

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

TEST(STR, SPLITTEST) {
  std::string data = "1\t2\t3";
  std::vector<std::string> res = SplitToStringVec(data, "\t");
  std::vector<std::string> expected = {"1", "2", "3"};
  EXPECT_EQ(res, expected);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}