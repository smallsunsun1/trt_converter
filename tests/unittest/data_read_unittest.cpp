#include "gtest/gtest.h"
#include <fstream>
#include <iostream>

TEST(FIEL_READ, SIMPLE_READ) {
  static const std::string filename = "./data/f1.txt";
  std::ifstream file(filename, std::ios::in);
  std::istreambuf_iterator<char> begin(file);
  std::istreambuf_iterator<char> end;
  std::string data(begin, end);
  std::string target_data = "1, 2, 3, 4\n5, 6, 7, 8";
  EXPECT_EQ(data, target_data);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}