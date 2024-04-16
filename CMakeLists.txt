cmake_minimum_required(VERSION 3.14)

project(gpuswap LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
set(PROTOBUF_IMPORT_DIRS "${CMAKE_SOURCE_DIR}/")

# PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS ./proto/cudarpc.proto)
# PROTOBUF_GENERATE_GRPC_CPP(GRPC_SRCS GRPC_HDRS ./proto/cudarpc.proto)

set(VENDOR_DIR common/vendor)

add_subdirectory(${VENDOR_DIR}/spdlog)
add_subdirectory(${VENDOR_DIR}/zeromq)
add_subdirectory(${VENDOR_DIR}/zeromqcpp)

include_directories(${PROTOBUF_INCLUDE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${ZEROMQ_INCLUDE_DIRS})
include_directories(${SPDLOG_INCLUDE_DIRS})
include_directories(${ZEROMQCPP_INCLUDE_DIRS})
include_directories(common/include)

include(FindProtobuf)
find_package(Protobuf REQUIRED)
# find_package(CUDAToolkit)
include_directories(${PROTOBUF_IMPORT_DIRS})
PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS ./proto/cudarpc.proto ./proto/signal.proto)

# find_package(Threads)
# find_package(protobuf REQUIRED)
# find_package(gRPC REQUIRED)

set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)

if("${BUILD_TARGET}" STREQUAL "client")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC" )
endif()

if("${BUILD_TARGET}" STREQUAL "server")
    find_package(Torch REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
endif()

add_library(rpc-library ${PROTO_HDRS} ${PROTO_SRCS})
target_link_libraries(rpc-library
        protobuf::libprotobuf
        )

file(GLOB_RECURSE ZMQ_UTIL_SRC common/include/zmq/*.cpp)
file(GLOB_RECURSE ZMQ_UTIL_HEADER common/include/zmq/*.hpp)
add_library(zmq-lib STATIC ${ZMQ_UTIL_HEADER} ${ZMQ_UTIL_SRC})
ADD_DEPENDENCIES(zmq-lib zeromq zeromqcpp spdlog) 

add_subdirectory(src)