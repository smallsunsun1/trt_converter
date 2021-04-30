#include "trt_converter/common/env.h"

#include <mutex>

#include "NvUtils.h"

namespace sss {

nvinfer1::Dims ToDims(const std::vector<int>& shape) {
  nvinfer1::Dims dims;
  dims.nbDims = shape.size();
  for (size_t i = 0; i < shape.size(); ++i) {
    dims.d[i] = shape[i];
  }
  return dims;
}

bool SetupInference(TRTInferenceEnvironment& env, const InferenceOptions& options) {
  for (int i = 0; i < options.streams; ++i) {
    env.contexts.emplace_back(env.engine->createExecutionContext());
    env.bindings_.emplace_back(new Bindings);
  }
  if (env.profiler) {
    env.contexts.front()->setProfiler(env.profiler.get());
  }
  const uint32_t n_opt_profilers = env.engine->getNbOptimizationProfiles();
  const uint32_t n_bindings = env.engine->getNbBindings();
  const uint32_t bindings_in_profile = n_opt_profilers > 0 ? n_bindings / n_opt_profilers : 0;
  const uint32_t end_binding_index = bindings_in_profile ? bindings_in_profile : env.engine->getNbBindings();
  if (n_opt_profilers > 1) {
    // TODO(jhsun), Need to add a logger here.
  }
  for (uint32_t b = 0; b < end_binding_index; ++b) {
    if (env.engine->bindingIsInput(b)) {
      auto dims = env.contexts.front()->getBindingDimensions(b);
      const bool is_scalar = dims.nbDims == 0;
      const bool is_dynamic_input = std::any_of(dims.d, dims.d + dims.nbDims, [](int dim) { return dim == -1; }) || env.engine->isShapeBinding(b);
      if (is_dynamic_input) {
        auto shape = options.shapes.find(env.engine->getBindingName(b));
        std::vector<int> static_dims;
        constexpr int kDefaultDimension = 1;
        if (shape == options.shapes.end()) {
          if (env.engine->isShapeBinding(b)) {
            if (is_scalar) {
              static_dims.push_back(1);
            } else {
              static_dims.resize(dims.d[0]);
              std::fill(static_dims.begin(), static_dims.end(), kDefaultDimension);
            }
          } else {
            static_dims.resize(dims.nbDims);
            std::transform(dims.d, dims.d + dims.nbDims, static_dims.begin(),
                           [&](int dimension) { return dimension >= 0 ? dimension : kDefaultDimension; });
          }
        } else {
          static_dims = shape->second;
        }
        for (auto& c : env.contexts) {
          if (env.engine->isShapeBinding(b)) {
            if (!c->setInputShapeBinding(b, static_dims.data())) {
              return false;
            }
          } else {
            if (!c->setBindingDimensions(b, ToDims(static_dims))) {
              return false;
            }
          }
        }
      }
    }
  }
  for (uint32_t b = 0; b < end_binding_index; ++b) {
    const auto dims = env.contexts.front()->getBindingDimensions(b);
    const auto vecDim = env.engine->getBindingVectorizedDim(b);
    const auto comps = env.engine->getBindingComponentsPerElement(b);
    const auto dataType = env.engine->getBindingDataType(b);
    const auto strides = env.contexts.front()->getStrides(b);
    const int batch = env.engine->hasImplicitBatchDimension() ? options.batch : 1;
    const auto vol = Volume(dims, strides, vecDim, comps, batch);
    const auto name = env.engine->getBindingName(b);
    const auto isInput = env.engine->bindingIsInput(b);
    for (auto& bindings : env.bindings_) {
      const auto input = options.inputs.find(name);
      if (isInput && input != options.inputs.end()) {
        bindings->AddBinding(b, name, isInput, vol, dataType, input->second);
      } else {
        bindings->AddBinding(b, name, isInput, vol, dataType);
      }
    }
  }
  return true;
}

namespace {

using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

//!
//! \struct SyncStruct
//! \brief Threads synchronization structure
//!
struct SyncStruct {
  std::mutex mutex;
  TRTCudaStream mainStream;
  TRTCudaEvent gpuStart;
  TimePoint cpuStart{};
  int sleep{0};
};

struct Enqueue {
  explicit Enqueue(nvinfer1::IExecutionContext& context, void** buffers) : context(context), buffers(buffers) {}
  nvinfer1::IExecutionContext& context;
  void** buffers{};
};

class EnqueueImplicit : Enqueue {
 public:
  EnqueueImplicit(nvinfer1::IExecutionContext& context, void** buffers, uint32_t batch) : Enqueue(context, buffers), batch(batch) {}
  void operator()(TRTCudaStream& stream) const { context.enqueue(batch, buffers, stream.Get(), nullptr); }

 private:
  uint32_t batch;
};

class EnqueueExplicit : Enqueue {
 public:
  EnqueueExplicit(nvinfer1::IExecutionContext& context, void** buffers) : Enqueue(context, buffers) {}
  void operator()(TRTCudaStream& stream) const { context.enqueueV2(buffers, stream.Get(), nullptr); }
};

class EnqueueGraph {
 public:
  explicit EnqueueGraph(TRTCudaGraph& graph) : graph(graph) {}
  void operator()(TRTCudaStream& stream) const { graph.Launch(stream); }
  TRTCudaGraph& graph;
};

using EnqueueFunction = std::function<void(TRTCudaStream&)>;

enum class StreamType : int { kINPUT = 0, kCOMPUTE = 1, kOUTPUT = 2, kNUM = 3 };

enum class EventType : int { kINPUT_S = 0, kINPUT_E = 1, kCOMPUTE_S = 2, kCOMPUTE_E = 3, kOUTPUT_S = 4, kOUTPUT_E = 5, kNUM = 6 };
using MultiStream = std::array<TRTCudaStream, static_cast<int>(StreamType::kNUM)>;

using MultiEvent = std::array<std::unique_ptr<TRTCudaEvent>, static_cast<int>(EventType::kNUM)>;

using EnqueueTimes = std::array<TimePoint, 2>;

class Iterator {
 public:
 private:
  void MoveNext() { next_ = depth_ - 1 - next_; }
  TRTCudaStream& GetStream(StreamType t) { return stream_[static_cast<int>(t)]; }
  TRTCudaEvent& GetEvent(EventType t) { return *events_[next_][static_cast<int>(t)]; }
  void Record(EventType e, StreamType s) { GetEvent(e).Record(GetStream(s)); }

  void RecordEnqueueTime() {
    enqueue_times_[next_][enqueue_start_] = std::chrono::high_resolution_clock::now();
    enqueue_start_ = 1 - enqueue_start_;
  }
  TimePoint GetEnqueueTime(bool start) { return enqueue_times_[next_][start ? 0 : 1]; }
  void Wait(EventType e, StreamType s) { GetStream(s).Wait(GetEvent(e)); }

  void CreateEnqueueFunction(const InferenceOptions& inference, nvinfer1::IExecutionContext& context, Bindings& bindings) {
    if (inference.batch) {
      enqueue_ = EnqueueFunction(EnqueueImplicit(context, bindings_.GetDeviceBuffers(), inference.batch));
    } else {
      enqueue_ = EnqueueFunction(EnqueueExplicit(context, bindings_.GetDeviceBuffers()));
      ;
    }
    if (inference.graph) {
      TRTCudaStream& stream = GetStream(StreamType::kCOMPUTE);
      enqueue_(stream);
      stream.Synchronize();
      graph_.BeginCapture(stream);
      enqueue_(stream);
      graph_.EndCapture(stream);
      enqueue_ = EnqueueFunction(EnqueueGraph(graph_));
    }
  }
  Bindings& bindings_;
  TRTCudaGraph graph_;
  EnqueueFunction enqueue_;
  uint32_t stream_id_ = 0;
  uint32_t next_ = 0;
  uint32_t depth_ = 0;
  std::vector<bool> active_;
  MultiStream stream_;
  std::vector<MultiEvent> events_;
  int enqueue_start_{0};
  std::vector<EnqueueTimes> enqueue_times_;
};

}  // namespace

}  // namespace sss