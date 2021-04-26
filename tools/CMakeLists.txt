find_package(protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)
set(grpc_path /usr/local/bin/ CACHE STRING /usr/local/bin/)
find_program(protoc_bin protoc ${grpc_path})
find_program(grpc_plugin grpc_cpp_plugin ${grpc_path})
find_program(grpc_python_plugin grpc_python_plugin ${grpc_path})

exec_program(${protoc_bin} ${CMAKE_CURRENT_SOURCE_DIR}/proto 
            ARGS "message.proto -I./ --plugin=protoc-gen-grpc=${grpc_plugin} --cpp_out=./ --grpc_out=./")
exec_program(${protoc_bin} ${CMAKE_CURRENT_SOURCE_DIR}/proto
            ARGS "message.proto -I./ --plugin=protoc-gen-grpc=${grpc_python_plugin} --python_out=./ --grpc_out=./")           

add_subdirectory(proto)

add_executable(client client.cpp)
target_link_libraries(client PRIVATE proto gRPC::grpc++ protobuf::libprotobuf)

add_executable(server server.cpp)
target_include_directories(server PUBLIC ${CMAKE_SOURCE_DIR}/third_party/async_lib)
target_link_libraries(server PRIVATE proto gRPC::grpc++ protobuf::libprotobuf async_context)
