#ifndef INCLUDE_TRT_CONVERTER_COMMON_MACRO_
#define INCLUDE_TRT_CONVERTER_COMMON_MACRO_

#include <cassert>
#include <iostream>

#include "cuda_runtime_api.h"

#define TRTC_NOEXCEPT noexcept

#define CUDA_CHECK(state)                         \
  do {                                            \
    if (state != cudaSuccess) {                   \
      assert(false && cudaGetErrorString(state)); \
    }                                             \
  } while (0)

#endif /* INCLUDE_TRT_CONVERTER_COMMON_MACRO_ */
