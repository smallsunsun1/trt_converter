file(GLOB source_file "*.cc")
add_library(proto ${source_file})
target_link_libraries(proto PUBLIC gRPC::grpc++ protobuf::libprotobuf z)