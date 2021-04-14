project(trt_conveter)
cmake_minimum_required(VERSION 3.15)


set(CMAKE_CUDA_HOST_COMPILER /usr/bin/gcc-9)
enable_language(CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(/home/sunjiahe/libs/TensorRT-7.2.3.4/include)

add_subdirectory(src)

add_executable(main main.cpp)