#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <chrono>
#include <vector>

int main(int argc, char** argv) {
    int src_device = 0;
    int dst_device = 1;
    if (argc > 1) {
        src_device = atoi(argv[1]);
    }

    if (argc > 2) {
        dst_device = atoi(argv[2]);
    }

    std::cout << "Src device " << src_device << " Dst device " << dst_device << std::endl;

    int count = 10;
    size_t sz = 1024 * 1024; // 1MB

    cudaError_t err;
    float* src_ptr;
    float* dst_ptr;

    cudaSetDevice(dst_device);
    err = cudaMalloc(&dst_ptr, sz);

    cudaSetDevice(src_device);
    err = cudaMalloc(&src_ptr, sz);
    cudaMemset(src_ptr, 0, sz);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEvent_t event;
    cudaEventCreate(&event);

    int is_able;
    cudaDeviceCanAccessPeer(&is_able, src_device, dst_device);
    std::cout << "Can access peer from " << src_device << " to " << dst_device << ": " << is_able << std::endl;

    if (is_able) {
        cudaDeviceEnablePeerAccess(dst_device, 0);
    }

    std::vector<int> elasped_vec;
    
    err = cudaMemcpyPeerAsync(dst_ptr, dst_device, src_ptr, src_device, sz, stream);
    cudaEventRecord(event, stream);
    cudaEventSynchronize(event);

    for (int i = 0; i < count; i++){
        auto start_t = std::chrono::system_clock::now();
        err = cudaMemcpyPeerAsync(dst_ptr, dst_device, src_ptr, src_device, sz, stream);
        cudaEventRecord(event, stream);
        cudaEventSynchronize(event);
        auto end_t = std::chrono::system_clock::now();
        auto mem_cpy_elasped = std::chrono::duration_cast<std::chrono::microseconds> (end_t - start_t).count();
        elasped_vec.push_back(mem_cpy_elasped);
    }

    int sum = 0;
    for (auto e: elasped_vec) sum += e;
    auto avg =  (float)sum / elasped_vec.size();

    std::cout << "Memcpy sz_in_kb " << sz / 1024 << " count " << count << " avg_elasped " << std::to_string(avg) << " throughput " << sz / 1024 / 1024 / avg * 1000 * 1000 << std::endl;
    // cudaDeviceSynchronize();
}
