#include "rpc_stream_server.h"

namespace sss {

BrainAsyncStreamServerImpl::BrainAsyncStreamServerImpl(async::HostContext* jobContext, int numThreads) {
  mpJobContext = jobContext;
  mThreads.resize(numThreads);
  mCompleteQueues.resize(numThreads);
  mShutDownState.resize(numThreads);
  mcomputeFunc = [this](Request* req, Response* res) { return DoBrainAnalysis(req, res); };
}

BrainAsyncStreamServerImpl::~BrainAsyncStreamServerImpl() {
  for (auto ss = mShutDownState.begin(); ss != mShutDownState.end(); ++ss) {
    std::lock_guard<std::mutex> lock((*ss)->Mmutex);
    (*ss)->mShutDown = true;
  }
  for (size_t i = 0; i < mCompleteQueues.size(); ++i) {
    mCompleteQueues[i]->Shutdown();
  }
  for (size_t i = 0; i < mThreads.size(); ++i) {
    mThreads[i].join();
  }
  for (auto cq = mCompleteQueues.begin(); cq != mCompleteQueues.end(); ++cq) {
    bool ok;
    void* tag;
    while ((*cq)->Next(&tag, &ok)) {
    }
  }
}

void BrainAsyncStreamServerImpl::Run(const std::string& serverAddress) {
  grpc::ServerBuilder builder;
  // 这里由于来回传输的数据会相对较大，这里采用压缩算法压缩数据
  builder.SetDefaultCompressionAlgorithm(GRPC_COMPRESS_GZIP);
  builder.AddListeningPort(serverAddress, grpc::InsecureServerCredentials());
  builder.RegisterService(&mService);
  for (int i = 0; i < mCompleteQueues.size(); ++i) {
    mCompleteQueues[i] = builder.AddCompletionQueue();
  }
  mpServer = builder.BuildAndStart();
  printf("Server listening on %s\n", serverAddress.c_str());
  // 这里我们提前建议好AsyncCallDataImpl数据结构来接受client的请求，默认先建立5000 x
  // num_complementionQueue数量的AsyncCallDataImpl
  constexpr int numCalls = 500;
  for (int connectionNum = 0; connectionNum < numCalls; ++connectionNum) {
    for (size_t i = 0; i < mCompleteQueues.size(); ++i) {
      mAsyncCallImplQueue.emplace_back(new AsyncCallDataImpl(&mService, mCompleteQueues[i].get(), mpJobContext, mcomputeFunc));
    }
  }
  for (size_t i = 0; i < mThreads.size(); ++i) {
    mShutDownState[i] = std::make_unique<PerThreadShutDownState>();
    mThreads[i] = std::thread([this, i]() { HandleRpcs(i); });
  }
}

void BrainAsyncStreamServerImpl::HandleRpcs(int i) {
  void* tag;
  bool ok;
  if (!mCompleteQueues[i]->Next(&tag, &ok)) {
    return;
  }
  RpcServerContext* ctx;
  std::mutex* mutexPtr = &mShutDownState[i]->Mmutex;
  do {
    ctx = reinterpret_cast<RpcServerContext*>(tag);
    mutexPtr->lock();
    if (mShutDownState[i]->mShutDown) {
      mutexPtr->unlock();
      return;
    }
  } while (mCompleteQueues[i]->DoThenAsyncNext(
      [&, ctx, ok, mutexPtr]() {
        ctx->lock();
        if (!ctx->RunNextStep(ok)) {
          ctx->Reset();
        }
        ctx->unlock();
        mutexPtr->unlock();
      },
      &tag, &ok, gpr_inf_future(GPR_CLOCK_REALTIME)));
}

}  // namespace sss

int main() {
  std::unique_ptr<sss::async::HostContext> context = sss::async::CreateCustomHostContext(4, 8);
  sss::BrainAsyncStreamServerImpl server(context.get(), 4);
  server.Run("localhost:50051");
  while (true) {
    std::this_thread::sleep_for(std::chrono::seconds(10));
  }
  return 0;
}
