#include <iostream>

#include "grpc++/grpc++.h"
#include "grpc/grpc.h"
#include "proto/message.grpc.pb.h"
#include "proto/message.pb.h"

namespace sss {

class SimpleClient {
 public:
  explicit SimpleClient(std::shared_ptr<grpc::ChannelInterface> channel) : stub_(RpcWork::NewStub(channel)) {}
  std::string RemoteCall(const std::string& request) {
    Request remote_request;
    remote_request.set_type(request);
    grpc::ClientContext context;
    Response result;
    grpc::Status status = stub_->RemoteCall(&context, remote_request, &result);
    if (status.ok()) {
      return result.type();
    } else {
      std::cerr << status.error_code() << " : " << status.error_message() << std::endl;
      return "RPC Failed!";
    }
  }
  void AsyncRemoteCall(const std::string& request) {
    AsyncCallData* req_data = new AsyncCallData();
    Request req;
    req.set_type(request);
    req_data->reader = stub_->PrepareAsyncRemoteCall(&req_data->context, req, queue_.get());
    req_data->reader->StartCall();
    req_data->reader->Finish(&req_data->response, &req_data->status, (void*)req_data);
  }
  void HandleAsyncWork() {
    void* tag;
    bool ok;
    while (queue_->Next(&tag, &ok)) {
      AsyncCallData* data = static_cast<AsyncCallData*>(tag);
      if (!data->status.ok()) {
        std::cerr << "error: " << data->status.error_code() << "  " << data->status.error_message();
      }
      delete data;
    }
  }

 private:
  struct AsyncCallData {
    Response response;
    grpc::ClientContext context;
    grpc::Status status;
    std::unique_ptr<grpc::ClientAsyncResponseReader<Response>> reader;
  };
  std::unique_ptr<grpc::CompletionQueue> queue_;
  std::unique_ptr<RpcWork::Stub> stub_;
};

// class SimpleStreamClient {
// public:
//     explicit SimpleStreamClient(std::shared_ptr<grpc::ChannelInterface> channel):stub_(RpcWork::NewStub(channel)) {}
//     Response StreamRemoteCall(Request req) {

//     }
// private:
//     std::unique_ptr<RpcWork::Stub> stub_;
// };

}  // namespace sss

int main(int argc, char* argv[]) {
  grpc::ChannelArguments arguments;
  grpc::EnableDefaultHealthCheckService(true);
  arguments.SetLoadBalancingPolicyName("grpclb");
  sss::SimpleClient client(grpc::CreateCustomChannel("localhost:50051", grpc::InsecureChannelCredentials(), arguments));
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < 10000000; ++i) {
    std::string name = "sss";
    std::string reply = client.RemoteCall(name);
    std::cout << "Greeter Received " << reply << std::endl;
  }
}