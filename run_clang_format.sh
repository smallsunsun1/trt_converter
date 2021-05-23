find include -name "*.h" | xargs -i -P20 clang-format -i {}
find include -name "*.cpp" | xargs -i -P20 clang-format -i {}

find src -name "*.h" | xargs -i -P20 clang-format -i {}
find src -name "*.cpp" | xargs -i -P20 clang-format -i {}

find tools -name "*.h" | xargs -i -P20 clang-format -i {}
find tools -name "*.cpp" | xargs -i -P20 clang-format -i {}

find tests -name "*.h" | xargs -i -P20 clang-format -i {}
find tests -name "*.cpp" | xargs -i -P20 clang-format -i {}
find tests -name "*.cu" | xargs -i -P20 clang-format -i {}