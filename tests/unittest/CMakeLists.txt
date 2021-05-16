add_executable(mnist_stream_unittest mnist_stream_unittest.cpp)
target_link_libraries(mnist_stream_unittest PUBLIC common ${GTEST_LIBRARIES})
target_include_directories(mnist_stream_unittest PRIVATE ${GTEST_INCLUDE_DIRS})
add_test(NAME test_mnist_stream COMMAND mnist_stream_unittest)

add_executable(data_read_unittest data_read_unittest.cpp)
target_link_libraries(data_read_unittest PUBLIC ${GTEST_LIBRARIES})
target_include_directories(data_read_unittest PRIVATE ${GTEST_INCLUDE_DIRS})
add_test(NAME test_data_read COMMAND data_read_unittest WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(str_function_unittest str_function_unittest.cpp)
target_link_libraries(str_function_unittest PUBLIC ${GTEST_LIBRARIES})
target_include_directories(str_function_unittest PRIVATE ${GTEST_INCLUDE_DIRS})
add_test(NAME test_str_function COMMAND str_function_unittest)

add_executable(logger_unittest logger_unittest.cpp)
target_link_libraries(logger_unittest PUBLIC ${GTEST_LIBRARIES} common)
target_include_directories(logger_unittest PRIVATE ${GTEST_INCLUDE_DIRS})
add_test(NAME test_logger COMMAND logger_unittest)

add_executable(filesystem_unittest filesystem_unittest.cpp)
target_link_libraries(filesystem_unittest PUBLIC ${GTEST_LIBRARIES} common)
target_include_directories(filesystem_unittest PRIVATE ${GTEST_INCLUDE_DIRS})
add_test(NAME test_filesystem COMMAND filesystem_unittest WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})