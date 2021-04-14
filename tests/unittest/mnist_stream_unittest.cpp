#include "trt_converter/calibrate/stream.h"

using namespace sss;

class MnistBatchStream: public DataStreamBase {
public:
    virtual void Reset() override {}
    virtual bool Next() override {return true;}
    virtual void Skip(uint32_t count) {}
    virtual float* GetBatch() {return nullptr;}
    virtual float* GetLabels() {return nullptr;}
    virtual int GetBatchesRead() {return 0;}
    virtual int GetBatchSize() const {return 0;}
    virtual nvinfer1::Dims GetDims() {return nvinfer1::Dims();}
};


int main(int argc, char* argv[]) {
    return 0;
}