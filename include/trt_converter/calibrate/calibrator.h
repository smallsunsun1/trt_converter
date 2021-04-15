#ifndef INCLUDE_TRT_CONVERTER_CALIBRATE_CALIBRATOR_
#define INCLUDE_TRT_CONVERTER_CALIBRATE_CALIBRATOR_

#include "NvInfer.h"

namespace sss {

template <typename Stream>
class EntropyCalibratorImpl {
    EntropyCalibratorImpl(Stream stream, int first_batch, const char* network_name, const char* input_blob_name, bool read_cache=true) {

    }
    int getBatchSize() const {}
    bool getBatch(void* bindings[], const char* names[], int nb_bindings){}
};

template <typename Stream>
class Int8EntropyCalibrator: public nvinfer1::IInt8EntropyCalibrator2 {
public:
     Int8EntropyCalibrator(
        Stream stream, int first_batch, const char* network_name, const char* input_blob_name, bool read_cache = true)
        : calib_impl_(stream, first_batch, network_name, input_blob_name, read_cache) {
    }

    int getBatchSize() const override
    {
        return calib_impl_.getBatchSize();
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) override
    {
        return calib_impl_.getBatch(bindings, names, nbBindings);
    }

    const void* readCalibrationCache(size_t& length) override
    {
        return calib_impl_.readCalibrationCache(length);
    }

    void writeCalibrationCache(const void* cache, size_t length) override
    {
        calib_impl_.writeCalibrationCache(cache, length);
    }
private:
    EntropyCalibratorImpl<Stream> calib_impl_;
};

}

#endif /* INCLUDE_TRT_CONVERTER_CALIBRATE_CALIBRATOR_ */
