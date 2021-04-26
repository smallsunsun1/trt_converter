#ifndef INCLUDE_TRT_CONVERTER_COMMON_COMMON_
#define INCLUDE_TRT_CONVERTER_COMMON_COMMON_

#include <iostream>
#include <NvInfer.h>

namespace sss {

inline std::ostream& operator<<(std::ostream& os, const nvinfer1::Dims& dims)
{
    os << "(";
    for (int i = 0; i < dims.nbDims; ++i)
    {
        os << (i ? ", " : "") << dims.d[i];
    }
    return os << ")";
}


}

#endif /* INCLUDE_TRT_CONVERTER_COMMON_COMMON_ */
