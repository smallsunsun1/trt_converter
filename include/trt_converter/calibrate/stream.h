#ifndef INCLUDE_TRT_CONVERTER_CALIBRATE_STREAM_
#define INCLUDE_TRT_CONVERTER_CALIBRATE_STREAM_

#include <cstdint>
#include "NvInfer.h"

namespace sss {

class DataStreamBase {
public:
    virtual void Reset() = 0;
    virtual bool Next() = 0;
    virtual void Skip(uint32_t count) = 0;
    virtual float* GetBatch() = 0;
    virtual float* GetLabels() = 0;
    virtual int GetBatchesRead() const = 0;
    virtual int GetBatchSize() const = 0;
    virtual nvinfer1::Dims GetDims() const = 0;
};

}

#endif /* INCLUDE_TRT_CONVERTER_CALIBRATE_STREAM_ */
