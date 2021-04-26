#ifndef TOOLS_RPC_STREAM_SERVER_
#define TOOLS_RPC_STREAM_SERVER_

#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "../third_party/async_lib/async/context/host_context.h"
#include "grpc/grpc.h"
#include "grpcpp/grpcpp.h"
#include "proto/message.grpc.pb.h"
#include "proto/message.pb.h"

namespace sss {

class RpcServerContext {
 public:
  RpcServerContext() {}
  void lock() { mMu.lock(); }
  void unlock() { mMu.unlock(); }
  virtual ~RpcServerContext() {}
  virtual bool RunNextStep(bool) = 0;  // 运行下一个Step
  virtual void Reset() = 0;            // 重置当前Context的状态
 private:
  std::mutex mMu;
};

class BrainAsyncStreamServerImpl {
 public:
  BrainAsyncStreamServerImpl(async::HostContext* jobContext, int numThreads);
  ~BrainAsyncStreamServerImpl();
  void Run(const std::string& serverAddress);
  grpc::Status DoBrainAnalysis(Request* mData, Response* result) { return grpc::Status::OK; }

 private:
  void HandleRpcs(int i);

  // template <typename BrainRequestData, typename BrainModuleResult>
  class AsyncCallDataImpl : public RpcServerContext {
   public:
    AsyncCallDataImpl(RpcWork::AsyncService* service, grpc::ServerCompletionQueue* queue, async::HostContext* jobContext,
                      std::function<grpc::Status(Request*, Response*)> invokeFunc)
        : mServerContext(new grpc::ServerContext),
          mNextState(&AsyncCallDataImpl::RequestDone),
          mpCompleteQueue(queue),
          mStream(mServerContext.get()),
          mpService(service),
          mpJobContext(jobContext),
          mInvokeMethod(invokeFunc) {
      // 异步接受Client请求
      mpService->RequestRemoteStreamCall(mServerContext.get(), &mStream, mpCompleteQueue, mpCompleteQueue, reinterpret_cast<void*>(this));
    }
    bool RunNextStep(bool ok) override { return (this->*mNextState)(ok); }
    void Reset() override {
      mServerContext.reset(new grpc::ServerContext);
      mReq = Request();
      mStream = grpc::ServerAsyncReaderWriter<Response, Request>(mServerContext.get());
      mNextState = &AsyncCallDataImpl::RequestDone;
      mpService->RequestRemoteStreamCall(mServerContext.get(), &mStream, mpCompleteQueue, mpCompleteQueue, reinterpret_cast<void*>(this));
    }
    ~AsyncCallDataImpl() override {}

   private:
    bool RequestDone(bool ok) {
      if (!ok) return false;
      mNextState = &AsyncCallDataImpl::ReadDone;
      mStream.Read(&mReq, reinterpret_cast<void*>(this));
      return true;
    }
    bool ReadDone(bool ok) {
      if (ok) {
        grpc::Status status = mInvokeMethod(&mReq, &mRes);
        mNextState = &AsyncCallDataImpl::WriteDone;
        mStream.Write(mRes, reinterpret_cast<void*>(this));
      } else {
        mNextState = &AsyncCallDataImpl::FinishDone;
        mStream.Finish(grpc::Status::OK, reinterpret_cast<void*>(this));
      }
      return true;
    }
    bool WriteDone(bool ok) {
      if (ok) {
        mNextState = &AsyncCallDataImpl::ReadDone;
        mStream.Read(&mReq, reinterpret_cast<void*>(this));
      } else {
        mNextState = &AsyncCallDataImpl::FinishDone;
        mStream.Finish(grpc::Status::OK, reinterpret_cast<void*>(this));
      }
      return true;
    }
    bool FinishDone(bool) { return false; }
    std::unique_ptr<grpc::ServerContext> mServerContext;
    grpc::ServerCompletionQueue* mpCompleteQueue;
    RpcWork::AsyncService* mpService;
    async::HostContext* mpJobContext;
    Request mReq;
    Response mRes;
    bool (AsyncCallDataImpl::*mNextState)(bool);
    std::function<grpc::Status(Request*, Response*)> mInvokeMethod;
    grpc::ServerAsyncReaderWriter<Response, Request> mStream;
  };
  struct PerThreadShutDownState {
    PerThreadShutDownState() : mShutDown(false) {}
    mutable std::mutex Mmutex;
    bool mShutDown;
  };
  // 脑部分相关数据
  async::HostContext* mpJobContext;

  std::function<grpc::Status(Request*, Response*)> mcomputeFunc;
  std::unique_ptr<grpc::Server> mpServer;
  RpcWork::AsyncService mService;
  // 这里设定CompleteQueues.size() == mThreads.size()
  std::vector<std::unique_ptr<grpc::ServerCompletionQueue>> mCompleteQueues;
  std::vector<std::thread> mThreads;
  std::vector<std::unique_ptr<PerThreadShutDownState>> mShutDownState;
  std::vector<std::unique_ptr<AsyncCallDataImpl>> mAsyncCallImplQueue;
};

}  // namespace sss

#endif /* TOOLS_RPC_STREAM_SERVER_ */
