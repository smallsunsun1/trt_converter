#include <cassert>
#include <iostream>
#include <mutex>
#include <thread>

#include "../third_party/async_lib/async/context/host_context.h"
#include "grpc++/grpc++.h"
#include "proto/message.grpc.pb.h"
#include "proto/message.pb.h"

namespace sss {

class SimpleServer {
 public:
  ~SimpleServer() {
    server_->Shutdown();
    server_queue_->Shutdown();
    void* tag;
    bool ok;
    while (server_queue_->Next(&tag, &ok)) {
      static_cast<CallData*>(tag)->Proceed();
    }
  }
  void Run(const std::string& host) {
    grpc::ServerBuilder builder;
    service_ = std::make_unique<RpcWork::AsyncService>();
    builder.AddListeningPort(host, grpc::InsecureServerCredentials());
    builder.RegisterService(service_.get());
    server_queue_ = builder.AddCompletionQueue();
    server_ = builder.BuildAndStart();
    std::cout << "Server Listen On " << host << std::endl;
    HandleRpcs();
  }

 private:
  struct CallData {
    CallData(RpcWork::AsyncService* service, grpc::ServerCompletionQueue* queue) : service(service), queue(queue), responder_(&ctx_), status(CREATE) {
      Proceed();
    }
    void Proceed() {
      if (status == CREATE) {
        status = PROCESS;
        service->RequestRemoteCall(&ctx_, &request, &responder_, queue, queue, (void*)this);
      } else if (status == PROCESS) {
        new CallData(service, queue);
        std::string prefix = "hello";
        response.set_type(prefix + request.type());
        status = FINISH;
        responder_.Finish(response, grpc::Status::OK, (void*)this);
      } else {
        delete this;
      }
    }
    RpcWork::AsyncService* service;
    grpc::ServerCompletionQueue* queue;
    grpc::ServerContext ctx_;
    Request request;
    Response response;
    grpc::ServerAsyncResponseWriter<Response> responder_;
    enum CallStatus { CREATE = 0, PROCESS = 1, FINISH = 2 };
    CallStatus status;
  };
  void HandleRpcs() {
    new CallData(service_.get(), server_queue_.get());
    void* tag;
    bool ok;
    while (true) {
      GPR_ASSERT(server_queue_->Next(&tag, &ok));
      GPR_ASSERT(ok);
      static_cast<CallData*>(tag)->Proceed();
    }
  }
  std::unique_ptr<RpcWork::AsyncService> service_;
  std::unique_ptr<grpc::ServerCompletionQueue> server_queue_;
  std::unique_ptr<grpc::Server> server_;
};

class RpcServerContext {
 public:
  RpcServerContext() = default;
  void Lock() { mu_.lock(); }
  void UnLock() { mu_.unlock(); }
  virtual ~RpcServerContext() {}
  virtual bool RunNextStep(bool) = 0;  // Run Next Step
  virtual void Reset() = 0;

 private:
  std::mutex mu_;
};

class SimpleStreamServer {
 public:
  SimpleStreamServer(async::HostContext* context, uint32_t num_threads) : run_context_(context) {
    threads_.resize(num_threads);
    server_queues_.resize(num_threads);
    shutdown_state_.resize(num_threads);
    compute_func_ = [this](Request* req, Response* res) -> grpc::Status { return DoRunFunc(req, res); };
  }
  ~SimpleStreamServer() {
    for (auto& state : shutdown_state_) {
      std::lock_guard<std::mutex> lock(state->mu);
      state->shut_down = true;
    }
    server_->Shutdown();
    for (size_t i = 0; i < server_queues_.size(); ++i) {
      server_queues_[i]->Shutdown();
    }
    for (auto& thread : threads_) {
      thread->join();
    }
    for (auto& cq : server_queues_) {
      bool ok;
      void* tag;
      while (cq->Next(&tag, &ok)) {
      }
    }
  }
  grpc::Status DoRunFunc(Request* req, Response* res) {
    res->set_type("Hello" + req->type());
    return grpc::Status::OK;
  }
  void Run(const std::string& server_address) {
    grpc::ServerBuilder builder;
    service_ = std::make_unique<RpcWork::AsyncService>();
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(service_.get());
    // Initialize Number Of CompletionQueue According To num_threads.
    for (size_t i = 0; i < server_queues_.size(); ++i) {
      server_queues_[i] = builder.AddCompletionQueue();
    }
    server_ = builder.BuildAndStart();
    std::cout << "Server Listen On " << server_address << std::endl;
    //  AsyncCallDataImpl size is qeual to kNumCalls * num_server_queues
    constexpr uint32_t kNumCalls = 1000;
    for (uint32_t connection_num = 0; connection_num < kNumCalls; ++connection_num) {
      for (size_t i = 0; i < server_queues_.size(); ++i) {
        async_call_impl_queue_.emplace_back(new AsyncCallDataImpl(service_.get(), server_queues_[i].get(), run_context_, compute_func_));
      }
    }
    for (size_t i = 0; i < threads_.size(); ++i) {
      shutdown_state_[i] = std::make_unique<PerThreadShutDownState>();
      threads_[i] = std::make_unique<std::thread>([this, i]() { HandleRpcs(i); });
    }
    server_->Wait();
  }
  void HandleRpcs(size_t idx) {
    void* tag;
    bool ok;
    std::cout << "Start To Get Tag From Queue!";
    if (!server_queues_[idx]->Next(&tag, &ok)) {
      return;
    }
    RpcServerContext* ctx;
    std::mutex* mutex_ptr = &shutdown_state_[idx]->mu;
    do {
      ctx = reinterpret_cast<RpcServerContext*>(tag);
      mutex_ptr->lock();
      if (shutdown_state_[idx]->shut_down) {
        mutex_ptr->unlock();
        return;
      }
    } while (server_queues_[idx]->DoThenAsyncNext(
        [&, ctx, ok, mutex_ptr]() {
          ctx->Lock();
          if (!ctx->RunNextStep(ok)) {
            ctx->Reset();
          }
          ctx->UnLock();
          mutex_ptr->unlock();
        },
        &tag, &ok, gpr_inf_future(GPR_CLOCK_REALTIME)));
  }

 private:
  class AsyncCallDataImpl : public RpcServerContext {
   public:
    AsyncCallDataImpl(RpcWork::AsyncService* service, grpc::ServerCompletionQueue* queue, async::HostContext* job_context,
                      std::function<grpc::Status(Request*, Response*)> invoke_func)
        : server_context_(new grpc::ServerContext),
          complete_queue_(queue),
          service_(service),
          job_context_(job_context),
          next_state_(&AsyncCallDataImpl::RequestDone),
          invoke_method_(invoke_func) {
      req = std::make_unique<Request>();
      res = std::make_unique<Response>();
      stream_ = std::make_unique<grpc::ServerAsyncReaderWriter<Response, Request>>(server_context_.get());
      service_->RequestRemoteStreamCall(server_context_.get(), stream_.get(), complete_queue_, complete_queue_, reinterpret_cast<void*>(this));
    }
    ~AsyncCallDataImpl() {}
    bool RunNextStep(bool ok) override { return (this->*next_state_)(ok); }
    void Reset() override {
      server_context_.reset(new grpc::ServerContext);
      req.reset(new Request);
      res.reset(new Response);
      stream_ = std::make_unique<grpc::ServerAsyncReaderWriter<Response, Request>>(server_context_.get());
      next_state_ = &AsyncCallDataImpl::RequestDone;
      service_->RequestRemoteStreamCall(server_context_.get(), stream_.get(), complete_queue_, complete_queue_, reinterpret_cast<void*>(this));
    }

   private:
    bool RequestDone(bool ok) {
      if (!ok) return false;
      next_state_ = &AsyncCallDataImpl::ReadDone;
      stream_->Read(req.get(), reinterpret_cast<void*>(this));
      return true;
    }
    bool ReadDone(bool ok) {
      if (ok) {
        grpc::Status status = invoke_method_(req.get(), res.get());
        next_state_ = &AsyncCallDataImpl::WriteDone;
        stream_->Write(*res, reinterpret_cast<void*>(this));
      } else {
        next_state_ = &AsyncCallDataImpl::FinishDone;
        stream_->Finish(grpc::Status::OK, reinterpret_cast<void*>(this));
      }
      return true;
    }
    bool WriteDone(bool ok) {
      if (ok) {
        next_state_ = &AsyncCallDataImpl::ReadDone;
        stream_->Read(req.get(), reinterpret_cast<void*>(this));
      } else {
        next_state_ = &AsyncCallDataImpl::FinishDone;
        stream_->Finish(grpc::Status::OK, reinterpret_cast<void*>(this));
      }
      return true;
    }
    bool FinishDone(bool) { return false; }
    std::unique_ptr<grpc::ServerContext> server_context_;
    grpc::ServerCompletionQueue* complete_queue_;
    RpcWork::AsyncService* service_;
    async::HostContext* job_context_;
    std::unique_ptr<Request> req;
    std::unique_ptr<Response> res;
    bool (AsyncCallDataImpl::*next_state_)(bool);
    std::function<grpc::Status(Request*, Response*)> invoke_method_;
    std::unique_ptr<grpc::ServerAsyncReaderWriter<Response, Request>> stream_;
  };
  struct PerThreadShutDownState {
    PerThreadShutDownState() : shut_down(false) {}
    mutable std::mutex mu;
    bool shut_down;
  };
  std::function<grpc::Status(Request*, Response*)> compute_func_;
  std::unique_ptr<RpcWork::AsyncService> service_;
  std::unique_ptr<grpc::Server> server_;
  async::HostContext* run_context_;
  std::vector<std::unique_ptr<grpc::ServerCompletionQueue>> server_queues_;
  std::vector<std::unique_ptr<std::thread>> threads_;
  std::vector<std::unique_ptr<PerThreadShutDownState>> shutdown_state_;
  std::vector<std::unique_ptr<AsyncCallDataImpl>> async_call_impl_queue_;
};

}  // namespace sss

int main(int argc, char* argv[]) {
  // sss::SimpleServer server;
  // server.Run("localhost:50051");

  std::unique_ptr<sss::async::HostContext> context = sss::async::CreateCustomHostContext(4, 8);
  sss::SimpleStreamServer server(context.get(), 4);
  server.Run("localhost:50051");
  return 0;
}