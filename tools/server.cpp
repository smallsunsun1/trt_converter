#include <iostream>

#include "grpc++/grpc++.h"
#include "proto/message.grpc.pb.h"
#include "proto/message.pb.h"

namespace sss {

class SimpleServer {
 public:
    ~SimpleServer(){
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
    CallData(RpcWork::AsyncService* service, grpc::ServerCompletionQueue* queue)
        : service(service), queue(queue), responder_(&ctx_), status(CREATE) {
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

}  // namespace sss

int main(int argc, char* argv[]) {
    sss::SimpleServer server;
    server.Run("localhost:50051"); 
    return 0; 
}