cmake_minimum_required(VERSION 3.14)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/target)

include_directories(../../include/client)
include_directories(/usr/local/cuda/include)

add_library(rtclient SHARED runtime.cpp)

target_link_libraries(rtclient
        zmq-lib
        zmq
        rpc-library
        )

add_library(mycuda SHARED driver.cpp)

target_link_libraries(mycuda
        zmq-lib
        zmq
        rpc-library
        )