#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <chrono>
#include <vector>
#include <future>

int main(int argc, char** argv) {
    int src_device = 0;
    if (argc > 1) {
        src_device = atoi(argv[1]);
    }

    int con = 4;
    if (argc > 2) {
        con = atoi(argv[2]);
    }

    auto gpu_count = 0;
    cudaGetDeviceCount(&gpu_count);
    cudaSetDevice(src_device);
    std::cout << "Src device " << src_device << " gpu count " << gpu_count << std::endl;

    for (int i = 0; i < gpu_count; i++) {
        if (i == src_device) continue;
        int is_able;
        cudaDeviceCanAccessPeer(&is_able, src_device, i);
        if (is_able) cudaDeviceEnablePeerAccess(i, 0);
    }

    int count = 10;
    size_t sz = 10 * 1024 * 1024; // 1MB

    cudaError_t err;
    float* src_ptr;

    cudaSetDevice(src_device);
    err = cudaMalloc(&src_ptr, sz);
    cudaMemset(src_ptr, 0, sz);

    auto transfer_func = [src_ptr](int src_device, int dst_device, size_t sz, int count) {
        cudaError_t err;
        float* dst_ptr;

        cudaSetDevice(dst_device);
        err = cudaMalloc(&dst_ptr, sz);

        cudaStream_t stream;
        cudaStreamCreate(&stream);
        cudaEvent_t event;
        cudaEventCreate(&event);
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
        return std::make_pair(dst_device, elasped_vec);
    };

    std::vector<std::future<std::pair<int, std::vector<int>>>> futures;
    int cur_con = 0;
    for (int i = 0; i < gpu_count; i++) {
        if (i == src_device) continue;
        futures.push_back(std::async(std::launch::async, transfer_func, src_device, i, sz, count));
        cur_con++;
        if (cur_con >= con) break;
    }

    for (auto& f: futures) {
        auto res = f.get();
        auto dst_device = res.first;
        auto elasped_vec = res.second;
        int sum = 0;
        for (auto e: elasped_vec) sum += e;
        auto avg =  (float)sum / elasped_vec.size();
        std::cout << "Memcpy from " << src_device << " to " << dst_device << " sz_in_kb " << sz / 1024 << " count " << count << " avg_elasped " << std::to_string(avg) << " throughput " << sz / 1024 / 1024 / avg * 1000 * 1000 << std::endl;
    }
}

// nvcc -o tt t.cpp -cudart shared -lpthread