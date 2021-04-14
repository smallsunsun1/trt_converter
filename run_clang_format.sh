find include -name "*.h" | xargs -i -P20 clang-format -i {}
find include -name "*.cpp" | xargs -i -P20 clang-format -i {}

find src -name "*.h" | xargs -i -P20 clang-format -i {}
find src -name "*.cpp" | xargs -i -P20 clang-format -i {}
