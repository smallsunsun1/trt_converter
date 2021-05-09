if (POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)
project(trt_conveter)
cmake_minimum_required(VERSION 3.15)


set(CMAKE_CUDA_HOST_COMPILER /usr/bin/gcc-9)
enable_language(CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61;-std=c++17;)
set(CMAKE_CXX_FLAGS "-fdiagnostics-color=always -fopenmp -fno-rtti -Wall -Wextra -Wno-deprecated -Wno-deprecated-declarations -Wno-invalid-offsetof")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")

find_package(Threads REQUIRED)
find_package(CUDA REQUIRED)
find_package(Glog REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(include)
include_directories(/home/sunjiahe/libs/TensorRT-7.2.3.4/include)
link_libraries(${CUDA_LIBRARIES} Threads::Threads glog::glog)

# include(cmake/fetch_google_benchmark.cmake)
# include(cmake/fetch_google_test.cmake)
# get_google_test()
# get_google_benchmark()

enable_testing()

add_subdirectory(third_party)
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(tools)

