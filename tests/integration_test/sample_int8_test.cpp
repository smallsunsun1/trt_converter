#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

#include "NvCaffeParser.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "cuda_runtime_api.h"
#include "trt_converter/calibrate/calibrator.h"
#include "trt_converter/calibrate/stream.h"
#include "trt_converter/common/argparser.h"
#include "trt_converter/common/buffer.h"
#include "trt_converter/common/common.h"
#include "trt_converter/common/logging.h"
#include "trt_converter/common/utils.h"

const std::string kSampleName = "TensorRT.sample_int8";

namespace sss {

class IBatchStream {
 public:
  virtual void reset(int firstBatch) = 0;
  virtual bool next() = 0;
  virtual void skip(int skipCount) = 0;
  virtual float* getBatch() = 0;
  virtual float* getLabels() = 0;
  virtual int getBatchesRead() const = 0;
  virtual int getBatchSize() const = 0;
  virtual nvinfer1::Dims getDims() const = 0;
};

class MNISTBatchStream : public IBatchStream {
 public:
  MNISTBatchStream(int batchSize, int maxBatches, const std::string& dataFile, const std::string& labelsFile,
                   const std::vector<std::string>& directories)
      : batch_size_{batchSize},
        max_batches_{maxBatches},
        dims_{3, {1, 28, 28}}  //!< We already know the dimensions of MNIST images.
  {
    readDataFile(LocateFile(dataFile, directories));
    readLabelsFile(LocateFile(labelsFile, directories));
  }

  void reset(int firstBatch) override { mBatchCount = firstBatch; }

  bool next() override {
    if (mBatchCount >= max_batches_) {
      return false;
    }
    ++mBatchCount;
    return true;
  }

  void skip(int skipCount) override { mBatchCount += skipCount; }

  float* getBatch() override { return mData.data() + (mBatchCount * batch_size_ * Volume(dims_)); }

  float* getLabels() override { return mLabels.data() + (mBatchCount * batch_size_); }

  int getBatchesRead() const override { return mBatchCount; }

  int getBatchSize() const override { return batch_size_; }
  nvinfer1::Dims getDims() const override {
    return nvinfer1::Dims{4, {batch_size_, dims_.d[0], dims_.d[1], dims_.d[2]}, {}};
  }

 private:
  void readDataFile(const std::string& dataFilePath) {
    std::ifstream file{dataFilePath.c_str(), std::ios::binary};

    int magicNumber, numImages, imageH, imageW;
    file.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
    // All values in the MNIST files are big endian.
    magicNumber = SwapEndianness(magicNumber);
    assert(magicNumber == 2051 && "Magic Number does not match the expected value for an MNIST image set");

    // Read number of images and dimensions
    file.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
    file.read(reinterpret_cast<char*>(&imageH), sizeof(imageH));
    file.read(reinterpret_cast<char*>(&imageW), sizeof(imageW));

    numImages = SwapEndianness(numImages);
    imageH = SwapEndianness(imageH);
    imageW = SwapEndianness(imageW);

    // The MNIST data is made up of unsigned bytes, so we need to cast to float and normalize.
    int numElements = numImages * imageH * imageW;
    std::vector<uint8_t> rawData(numElements);
    file.read(reinterpret_cast<char*>(rawData.data()), numElements * sizeof(uint8_t));
    mData.resize(numElements);
    std::transform(rawData.begin(), rawData.end(), mData.begin(),
                   [](uint8_t val) { return static_cast<float>(val) / 255.f; });
  }
  void readLabelsFile(const std::string& labelsFilePath) {
    std::ifstream file{labelsFilePath.c_str(), std::ios::binary};
    int magicNumber, numImages;
    file.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
    // All values in the MNIST files are big endian.
    magicNumber = SwapEndianness(magicNumber);
    assert(magicNumber == 2049 && "Magic Number does not match the expected value for an MNIST labels file");

    file.read(reinterpret_cast<char*>(&numImages), sizeof(numImages));
    numImages = SwapEndianness(numImages);

    std::vector<uint8_t> rawLabels(numImages);
    file.read(reinterpret_cast<char*>(rawLabels.data()), numImages * sizeof(uint8_t));
    mLabels.resize(numImages);
    std::transform(rawLabels.begin(), rawLabels.end(), mLabels.begin(),
                   [](uint8_t val) { return static_cast<float>(val); });
  }

  int batch_size_{0};
  int mBatchCount{0};  //!< The batch that will be read on the next invocation of next()
  int max_batches_{0};
  nvinfer1::Dims dims_{};
  std::vector<float> mData{};
  std::vector<float> mLabels{};
};

struct SampleInt8Params : public CaffeModelParams {
  uint32_t nb_cal_batches;
  uint32_t cal_batch_size;
  std::string network_name;
};

class SampleInt8 {
 public:
  template <typename T>
  using SampleUniquePtr = std::unique_ptr<T, InferDeleter>;
  SampleInt8(const SampleInt8Params& params) : params_(params), engine_(nullptr) {
    initLibNvInferPlugins(&Logger::Instance(), "");
  }
  bool Build(nvinfer1::DataType dataType);

  bool IsSupported(nvinfer1::DataType dataType);

  bool Infer(std::vector<float>& score, int firstScoreBatch, int nbScoreBatches);

  bool Teardown();

 private:
  SampleInt8Params params_;
  nvinfer1::Dims input_dims_;
  std::shared_ptr<nvinfer1::ICudaEngine> engine_;

  bool ConstructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
                        SampleUniquePtr<nvinfer1::INetworkDefinition>& network,
                        SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
                        SampleUniquePtr<nvcaffeparser1::ICaffeParser>& parser, nvinfer1::DataType dataType);

  bool ProcessInput(const BufferManager& buffers, const float* data);

  int CalculateScore(const BufferManager& buffers, float* labels, int batchSize, int outputSize, int threshold);
};

bool SampleInt8::Build(nvinfer1::DataType dataType) {
  SampleUniquePtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(Logger::Instance()));
  if (builder) {
    return false;
  }
  SampleUniquePtr<nvinfer1::INetworkDefinition> network(builder->createNetwork());
  if (network) {
    return false;
  }
  SampleUniquePtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
  if (!config) {
    return false;
  }
  auto parser = SampleUniquePtr<nvcaffeparser1::ICaffeParser>(nvcaffeparser1::createCaffeParser());
  if (!parser) {
    return false;
  }

  if ((dataType == nvinfer1::DataType::kINT8 && !builder->platformHasFastInt8()) ||
      (dataType == nvinfer1::DataType::kHALF && !builder->platformHasFastFp16())) {
    return false;
  }

  auto constructed = ConstructNetwork(builder, network, config, parser, dataType);
  if (!constructed) {
    return false;
  }

  assert(network->getNbInputs() == 1);
  input_dims_ = network->getInput(0)->getDimensions();
  assert(mInputDims.nbDims == 3);
  return true;
}

bool SampleInt8::IsSupported(nvinfer1::DataType dataType) {
  auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(Logger::Instance()));
  if (!builder) {
    return false;
  }

  if ((dataType == nvinfer1::DataType::kINT8 && !builder->platformHasFastInt8()) ||
      (dataType == nvinfer1::DataType::kHALF && !builder->platformHasFastFp16())) {
    return false;
  }

  return true;
}

bool SampleInt8::Teardown() {
  nvcaffeparser1::shutdownProtobufLibrary();
  return true;
}

bool SampleInt8::ConstructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
                                  SampleUniquePtr<nvinfer1::INetworkDefinition>& network,
                                  SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
                                  SampleUniquePtr<nvcaffeparser1::ICaffeParser>& parser, nvinfer1::DataType dataType) {
  engine_ = nullptr;
  const nvcaffeparser1::IBlobNameToTensor* blobNameToTensor =
      parser->parse(LocateFile(params_.prototxt_filename, params_.data_dirs).c_str(),
                    LocateFile(params_.weights_filename, params_.data_dirs).c_str(), *network,
                    dataType == nvinfer1::DataType::kINT8 ? nvinfer1::DataType::kFLOAT : dataType);
  for (auto& s : params_.output_tensor_names) {
    network->markOutput(*blobNameToTensor->find(s.c_str()));
  }

  // Calibrator life time needs to last until after the engine is built.
  std::unique_ptr<nvinfer1::IInt8Calibrator> calibrator;

  config->setAvgTimingIterations(1);
  config->setMinTimingIterations(1);
  config->setMaxWorkspaceSize(1.0_GiB);
  if (dataType == nvinfer1::DataType::kHALF) {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  }
  if (dataType == nvinfer1::DataType::kINT8) {
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
  }
  builder->setMaxBatchSize(params_.batch_size);

  if (dataType == nvinfer1::DataType::kINT8) {
    MNISTBatchStream calibrationStream(params_.cal_batch_size, params_.nb_cal_batches, "train-images-idx3-ubyte",
                                       "train-labels-idx1-ubyte", params_.data_dirs);
    calibrator.reset(new Int8EntropyCalibrator<MNISTBatchStream>(calibrationStream, 0, params_.network_name.c_str(),
                                                                 params_.input_tensor_names[0].c_str()));
    config->setInt8Calibrator(calibrator.get());
  }

  if (params_.dla_core >= 0) {
    // (builder.get(), config.get(), mParams.dlaCore);
    if (params_.batch_size > builder->getMaxDLABatchSize()) {
      LOG(ERROR) << "Requested batch size " << params_.batch_size << " is greater than the max DLA batch size of "
                 << builder->getMaxDLABatchSize() << ". Reducing batch size accordingly." << std::endl;
      return false;
    }
  }

  engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config), InferDeleter());
  if (!engine_) {
    return false;
  }
  return true;
}

bool SampleInt8::Infer(std::vector<float>& score, int firstScoreBatch, int nbScoreBatches) {
  float ms{0.0f};

  // Create RAII buffer manager object
  BufferManager buffers(engine_, params_.batch_size);

  auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
  if (!context) {
    return false;
  }

  MNISTBatchStream batchStream(params_.batch_size, nbScoreBatches + firstScoreBatch, "train-images-idx3-ubyte",
                               "train-labels-idx1-ubyte", params_.data_dirs);
  batchStream.skip(firstScoreBatch);

  nvinfer1::Dims outputDims = context->getEngine().getBindingDimensions(
      context->getEngine().getBindingIndex(params_.output_tensor_names[0].c_str()));
  int outputSize = Volume(outputDims);
  int top1{0}, top5{0};
  float totalTime{0.0f};

  while (batchStream.next()) {
    // Read the input data into the managed buffers
    assert(mParams.inputTensorNames.size() == 1);
    if (!ProcessInput(buffers, batchStream.getBatch())) {
      return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // Use CUDA events to measure inference time
    cudaEvent_t start, end;
    CHECK(cudaEventCreateWithFlags(&start, cudaEventBlockingSync));
    CHECK(cudaEventCreateWithFlags(&end, cudaEventBlockingSync));
    cudaEventRecord(start, stream);

    bool status = context->enqueue(params_.batch_size, buffers.getDeviceBindings().data(), stream, nullptr);
    if (!status) {
      return false;
    }

    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    totalTime += ms;

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    CHECK(cudaStreamDestroy(stream));

    top1 += CalculateScore(buffers, batchStream.getLabels(), params_.batch_size, outputSize, 1);
    top5 += CalculateScore(buffers, batchStream.getLabels(), params_.batch_size, outputSize, 5);

    if (batchStream.getBatchesRead() % 100 == 0) {
      LOG(INFO) << "Processing next set of max 100 batches" << std::endl;
    }
  }

  int imagesRead = (batchStream.getBatchesRead() - firstScoreBatch) * params_.batch_size;
  score[0] = float(top1) / float(imagesRead);
  score[1] = float(top5) / float(imagesRead);

  LOG(INFO) << "Top1: " << score[0] << ", Top5: " << score[1] << std::endl;
  LOG(INFO) << "Processing " << imagesRead << " images averaged " << totalTime / imagesRead << " ms/image and "
            << totalTime / batchStream.getBatchesRead() << " ms/batch." << std::endl;

  return true;
}

bool SampleInt8::ProcessInput(const BufferManager& buffers, const float* data)
{
    // Fill data buffer
    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(params_.input_tensor_names[0]));
    std::memcpy(hostDataBuffer, data, params_.batch_size * Volume(input_dims_) * sizeof(float));
    return true;
}

int SampleInt8::CalculateScore(
    const BufferManager& buffers, float* labels, int batchSize, int outputSize, int threshold)
{
    float* probs = static_cast<float*>(buffers.getHostBuffer(params_.output_tensor_names[0]));

    int success = 0;
    for (int i = 0; i < batchSize; i++)
    {
        float *prob = probs + outputSize * i, correct = prob[(int) labels[i]];

        int better = 0;
        for (int j = 0; j < outputSize; j++)
        {
            if (prob[j] >= correct)
            {
                better++;
            }
        }
        if (better <= threshold)
        {
            success++;
        }
    }
    return success;
}

SampleInt8Params initializeSampleParams(const Args& args, int batchSize)
{
    SampleInt8Params params;
    // Use directories provided by the user, in addition to default directories.
    params.data_dirs = args.data_dirs;
    params.data_dirs.emplace_back("data/mnist/");
    params.data_dirs.emplace_back("int8/mnist/");
    params.data_dirs.emplace_back("samples/mnist/");
    params.data_dirs.emplace_back("data/samples/mnist/");
    params.data_dirs.emplace_back("data/int8/mnist/");
    params.data_dirs.emplace_back("data/int8_samples/mnist/");

    params.batch_size = batchSize;
    params.dla_core = args.use_dla_core;
    params.nb_cal_batches = 10;
    params.cal_batch_size = 50;
    params.input_tensor_names.push_back("data");
    params.output_tensor_names.push_back("prob");
    params.prototxt_filename = "deploy.prototxt";
    params.weights_filename = "mnist_lenet.caffemodel";
    params.network_name = "mnist";
    return params;
}

void PrintHelpInfo()
{
    std::cout << "Usage: ./sample_int8 [-h or --help] [-d or --datadir=<path to data directory>] "
                 "[--useDLACore=<int>]"
              << std::endl;
    std::cout << "--help, -h      Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories."
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
    std::cout << "batch=N         Set batch size (default = 32)." << std::endl;
    std::cout << "start=N         Set the first batch to be scored (default = 16). All batches before this batch will "
                 "be used for calibration."
              << std::endl;
    std::cout << "score=N         Set the number of batches to be scored (default = 1800)." << std::endl;
}

int main(int argc, char** argv)
{
    if (argc >= 2 && (!strncmp(argv[1], "--help", 6) || !strncmp(argv[1], "-h", 2)))
    {
        PrintHelpInfo();
        return 0;
    }

    // By default we score over 57600 images starting at 512, so we don't score those used to search calibration
    int batchSize = 32;
    int firstScoreBatch = 16;
    int nbScoreBatches = 1800;

    // Parse extra arguments
    for (int i = 1; i < argc; ++i)
    {
        if (!strncmp(argv[i], "batch=", 6))
        {
            batchSize = atoi(argv[i] + 6);
        }
        else if (!strncmp(argv[i], "start=", 6))
        {
            firstScoreBatch = atoi(argv[i] + 6);
        }
        else if (!strncmp(argv[i], "score=", 6))
        {
            nbScoreBatches = atoi(argv[i] + 6);
        }
    }

    if (batchSize > 128)
    {
        LOG(ERROR) << "Please provide batch size <= 128" << std::endl;
        return EXIT_FAILURE;
    }

    if ((firstScoreBatch + nbScoreBatches) * batchSize > 60000)
    {
        LOG(ERROR) << "Only 60000 images available" << std::endl;
        return EXIT_FAILURE;
    }

    Args args;
    // ParseArgs(args, argc, argv);

    // SampleInt8 sample(initializeSampleParams(args, batchSize));

    // auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);

    // sample::gLogger.reportTestStart(sampleTest);

    // sample::gLogInfo << "Building and running a GPU inference engine for INT8 sample" << std::endl;

    // std::vector<std::string> dataTypeNames = {"FP32", "FP16", "INT8"};
    // std::vector<std::string> topNames = {"Top1", "Top5"};
    // std::vector<nvinfer1::DataType> dataTypes = {nvinfer1::DataType::kFLOAT, 
                                                  // nvinfer1::DataType::kHALF, nvinfer1::DataType::kINT8};
    // std::vector<std::vector<float>> scores(3, std::vector<float>(2, 0.0f));
    // for (size_t i = 0; i < dataTypes.size(); i++)
    // {
    //     sample::gLogInfo << dataTypeNames[i] << " run:" << nbScoreBatches << " batches of size " << batchSize
    //                      << " starting at " << firstScoreBatch << std::endl;

    //     if (!sample.build(dataTypes[i]))
    //     {
    //         if (!sample.isSupported(dataTypes[i]))
    //         {
    //             sample::gLogWarning << "Skipping " << dataTypeNames[i]
    //                                 << " since the platform does not support this data type." << std::endl;
    //             continue;
    //         }
    //         return sample::gLogger.reportFail(sampleTest);
    //     }
    //     if (!sample.infer(scores[i], firstScoreBatch, nbScoreBatches))
    //     {
    //         return sample::gLogger.reportFail(sampleTest);
    //     }
    // }

    // auto isApproximatelyEqual = [](float a, float b, double tolerance) { return (std::abs(a - b) <= tolerance); };
    // const double tolerance{0.01};
    // const double goldenMNIST{0.99};

    // if ((scores[0][0] < goldenMNIST) || (scores[0][1] < goldenMNIST))
    // {
    //     sample::gLogError << "FP32 accuracy is less than 99%: Top1 = " << scores[0][0] << ", Top5 = " << scores[0][1]
    //                       << "." << std::endl;
    //     return sample::gLogger.reportFail(sampleTest);
    // }

    // for (unsigned i = 0; i < topNames.size(); i++)
    // {
    //     for (unsigned j = 1; j < dataTypes.size(); j++)
    //     {
    //         if (scores[j][i] != 0.0f && !isApproximatelyEqual(scores[0][i], scores[j][i], tolerance))
    //         {
    //             sample::gLogError << "FP32(" << scores[0][i] << ") and " << dataTypeNames[j] << "(" << scores[j][i]
    //                               << ") " << topNames[i] << " accuracy differ by more than " << tolerance << "."
    //                               << std::endl;
    //             return sample::gLogger.reportFail(sampleTest);
    //         }
    //     }
    // }

    // if (!sample.Teardown())
    // {
    //     return sample::gLogger.reportFail(sampleTest);
    // }

    // return sample::gLogger.reportPass(sampleTest);
    return 0;
}


}  // namespace sss
