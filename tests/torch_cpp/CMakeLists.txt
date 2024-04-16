cmake_minimum_required(VERSION 3.14)

project(main CUDA CXX)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(/usr/local/cuda/include)

add_library(wapper SHARED api_wapper.cpp)

add_executable(main main.cpp)
target_compile_features(main PUBLIC cxx_range_for)
target_link_libraries(main ${TORCH_LIBRARIES})
target_link_libraries(main wapper)
set_property(TARGET main PROPERTY CXX_STANDARD 14)