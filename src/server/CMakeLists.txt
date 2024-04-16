cmake_minimum_required(VERSION 3.14)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/target)

include_directories(../../include/server)
# include_directories(/opt/conda/lib/python3.8/site-packages/torch/include/)
# include_directories(/opt/conda/lib/python3.8/site-packages/torch/include/torch/csrc/api/include)

include_directories(/usr/local/cuda/include)
add_library(wapper SHARED kernel_lookup.cpp)

add_executable(server server.cpp)

target_link_libraries(server
        zmq-lib
        zmq
        rpc-library
        cuda
        cublas
        cudnn
        curand
        ${TORCH_LIBRARIES}
        wapper
        # torch
        # torch_cuda
    )

# ADD_DEPENDENCIES(server zmq-lib zeromq zeromqcpp)