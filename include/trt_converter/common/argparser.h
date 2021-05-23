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

// inline bool parseArgs(Args& args, int32_t argc, char* argv[]) {
//   while (1) {
//     int32_t arg;
//     static struct Options long_options[] = {
//         {"help", no_argument, 0, 'h'},        {"datadir", required_argument, 0, 'd'},
//         {"int8", no_argument, 0, 'i'},        {"fp16", no_argument, 0, 'f'},
//         {"useILoop", no_argument, 0, 'l'},    {"saveEngine", required_argument, 0, 's'},
//         {"loadEngine", no_argument, 0, 'o'},  {"useDLACore", required_argument, 0, 'u'},
//         {"batch", required_argument, 0, 'b'}, {nullptr, 0, nullptr, 0}};
//     int32_t option_index = 0;
//     arg = getopt_long(argc, argv, "hd:iu", long_options, &option_index);
//     if (arg == -1) {
//       break;
//     }

//     switch (arg) {
//       case 'h':
//         args.help = true;
//         return true;
//       case 'd':
//         if (optarg) {
//           args.dataDirs.push_back(optarg);
//         } else {
//           std::cerr << "ERROR: --datadir requires option argument" << std::endl;
//           return false;
//         }
//         break;
//       case 's':
//         if (optarg) {
//           args.saveEngine = optarg;
//         }
//         break;
//       case 'o':
//         if (optarg) {
//           args.load_engine = optarg;
//         }
//         break;
//       case 'i':
//         args.runInInt8 = true;
//         break;
//       case 'f':
//         args.runInFp16 = true;
//         break;
//       case 'l':
//         args.useILoop = true;
//         break;
//       case 'u':
//         if (optarg) {
//           args.useDLACore = std::stoi(optarg);
//         }
//         break;
//       case 'b':
//         if (optarg) {
//           args.batch = std::stoi(optarg);
//         }
//         break;
//       default:
//         return false;
//     }
//   }
//   return true;
// }

}  // namespace sss

#endif /* INCLUDE_TRT_CONVERTER_COMMON_ARGPARSER_ */
