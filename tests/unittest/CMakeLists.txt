add_executable(mnist_stream_unittest mnist_stream_unittest.cpp)
target_link_libraries(mnist_stream_unittest PUBLIC common ${GTEST_LIBRARIES})
target_include_directories(mnist_stream_unittest PRIVATE ${GTEST_INCLUDE_DIRS})
add_test(NAME test_mnist_stream COMMAND mnist_stream_unittest)

add_executable(data_read_unittest data_read_unittest.cpp)
target_link_libraries(data_read_unittest PUBLIC ${GTEST_LIBRARIES})
target_include_directories(data_read_unittest PRIVATE ${GTEST_INCLUDE_DIRS})
add_test(NAME test_data_read COMMAND data_read_unittest WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})