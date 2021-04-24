find_package(protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_program(protoc_bin protoc /usr/local/bin/)
find_program(grpc_plugin grpc_cpp_plugin /usr/local/bin/)

exec_program(${protoc_bin} ${CMAKE_CURRENT_SOURCE_DIR}/proto ARGS "message.proto -I./ --plugin=protoc-gen-grpc=${grpc_plugin} --cpp_out=./ --grpc_out=./")

add_subdirectory(proto)

add_executable(client client.cpp)
target_link_libraries(client PRIVATE proto gRPC::grpc++ protobuf::libprotobuf)

add_executable(server server.cpp)
target_link_libraries(server PRIVATE proto gRPC::grpc++ protobuf::libprotobuf)