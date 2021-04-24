#include <iostream>
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

class SimpleStreamServer {
 public:
  SimpleStreamServer(async::HostContext* context) : run_context_(context) {}
  ~SimpleStreamServer() {}
  void Run(const std::string& server_address, uint32_t num_threads = std::thread::hardware_concurrency() * 2) {
    grpc::ServerBuilder builder;
    service_ = std::make_unique<RpcWork::AsyncService>();
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(service_.get());
    // Initialize Number Of CompletionQueue According To num_threads.
    for (uint32_t i = 0; i < num_threads; ++i) {
      server_queues_[i] = builder.AddCompletionQueue();
    }
    builder.BuildAndStart();
    std::cout << "Server Listen On " << server_address << std::endl;
    for (uint32_t i = 0; i < num_threads; ++i) {
      run_context_->EnqueueBlockingWork([this, i]() { HandleRpcs(static_cast<size_t>(i)); });
    }
  }
  void HandleRpcs(size_t idx) {
    grpc::ServerCompletionQueue* queue = server_queues_[idx].get();
    void* tag;
    bool ok;
    while (true) {
      GPR_ASSERT(queue->Next(&tag, &ok));
      GPR_ASSERT(ok);
      static_cast<AsyncCallData*>(tag)->Proceed();
    }
  }

 private:
  struct AsyncCallData {
    void Proceed() {}
  };
  std::unique_ptr<RpcWork::AsyncService> service_;
  std::unique_ptr<grpc::Server> server_;
  async::HostContext* run_context_;
  std::vector<std::unique_ptr<grpc::ServerCompletionQueue>> server_queues_;
};

}  // namespace sss

int main(int argc, char* argv[]) {
  sss::SimpleServer server;
  server.Run("localhost:50051");
  return 0;
}