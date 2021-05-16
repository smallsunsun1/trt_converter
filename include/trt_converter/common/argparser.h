#ifndef INCLUDE_TRT_CONVERTER_COMMON_ARGPARSER_
#define INCLUDE_TRT_CONVERTER_COMMON_ARGPARSER_

#include <iostream>
#include <string>
#include <vector>

namespace sss {

struct ModelParams {
  int32_t batch_size = 1;              //!< Number of inputs in a batch
  int32_t dla_core = -1;               //!< Specify the DLA core to run network on.
  bool int8 = false;                   //!< Allow runnning the network in Int8 mode.
  bool fp16 = false;                   //!< Allow running the network in FP16 mode.
  std::vector<std::string> data_dirs;  //!< Directory paths where sample data files are stored
  std::vector<std::string> input_tensor_names;
  std::vector<std::string> output_tensor_names;
};

struct CaffeModelParams : public ModelParams {
  std::string prototxt_filename;
  std::string weights_filename;
  std::string mean_filename;
};

struct ONNXModelParams : public ModelParams {
  std::string onnx_filename;
};

struct Args {
  bool run_in_int8 = false;
  bool run_in_fp16 = false;
  bool help = false;
  int32_t use_dla_core = false;
  int32_t batch = 1;
  std::vector<std::string> data_dirs;
  std::string save_engine;
  std::string load_engine;
  bool use_iloop = false;
};

}  // namespace sss

#endif /* INCLUDE_TRT_CONVERTER_COMMON_ARGPARSER_ */
