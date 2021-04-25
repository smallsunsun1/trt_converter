import grpc
import message_pb2
import message_pb2_grpc


def run(host_name: str):
    channel = grpc.insecure_channel(host_name)
    stub = message_pb2_grpc.RpcWorkStub(channel)
    response = stub.RemoteCall(message_pb2.Request(type="test"))
    print("Greeter client received: " + response.type)

if __name__ == "__main__":
    host_name = "localhost:50051"
    run(host_name)
