project(trt_conveter)
cmake_minimum_required(VERSION 3.15)


set(CMAKE_CUDA_HOST_COMPILER /usr/bin/gcc-9)
enable_language(CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "-fdiagnostics-color=auto -fno-rtti -Wall -Wextra -Wno-deprecated-copy -Wno-deprecated-declarations")

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(include)
include_directories(/home/sunjiahe/libs/TensorRT-7.2.3.4/include)

# include(cmake/fetch_google_benchmark.cmake)
include(cmake/fetch_google_test.cmake)
get_google_test()
# get_google_benchmark()


add_subdirectory(src)
add_subdirectory(tests)

add_executable(main main.cpp)