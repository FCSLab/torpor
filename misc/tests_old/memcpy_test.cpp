#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <chrono>
#include <vector>
#include <cstring>

// void cuda_memcpy_thread(int device, float* host_ptr, size_t sz) {
//     cudaSetDevice(device);
//     float* device_ptr;
//     cudaMalloc(&device_ptr, sz);
// }

int main(int argc, char** argv) {
    int con_num = 1;
    if (argc > 1) {
        con_num = atoi(argv[1]);
    }

    int count = 10;
    if (argc > 2) {
        count = atoi(argv[2]);
    }

    size_t sz = 1024 * 1024 * 100 / count;

    cudaError_t err;
    void* host_ptr;
    void* cuda_ptrs[con_num];
    err = cudaMallocHost(&host_ptr, sz * count);
    memset(host_ptr, 0, sz * count);

    for (int i = 0; i < con_num; i++){
        cudaSetDevice(i);
        err = cudaMalloc(&cuda_ptrs[i],  sz * count);
    }

    cudaStream_t stream[con_num];
    for (int i = 0; i < con_num; i++){
        cudaStreamCreate(&stream[i]);
    }
    cudaEvent_t event;
    cudaEventCreate(&event);

    std::vector<int> elasped_vec;

    auto start_first_t = std::chrono::system_clock::now();
    for (int i = 0; i < count; i++){
        for (int j = 0; j < con_num; j++){
            cudaSetDevice(j);
            err = cudaMemcpyAsync(cuda_ptrs[j] + i * sz, host_ptr + i * sz, sz, cudaMemcpyHostToDevice, stream[j]);
        }
    }
    cudaDeviceSynchronize();
    auto end_first_t = std::chrono::system_clock::now();
    std::cout << "Startup memcpy time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_first_t - start_first_t).count() << std::endl;

    for (int t = 0; t < 10; t++){
        auto start_t = std::chrono::system_clock::now();
        for (int i = 0; i < count; i++){
            for (int j = 0; j < con_num; j++){
                cudaSetDevice(j);
                err = cudaMemcpyAsync(cuda_ptrs[j] + i * sz, host_ptr + i * sz, sz, cudaMemcpyHostToDevice, stream[j]);
            }
        }
        cudaDeviceSynchronize();
        auto end_t = std::chrono::system_clock::now();
        auto elasped = std::chrono::duration_cast<std::chrono::microseconds>(end_t - start_t).count();
        elasped_vec.push_back(elasped);
    }
    auto end_last_t = std::chrono::system_clock::now();
    std::cout << "Test memcpy time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_last_t - end_first_t).count() << std::endl;

    int sum = 0;
    for (auto e: elasped_vec) sum += e;
    auto avg =  (float)sum / elasped_vec.size();

    std::cout << "Memcpy sz_in_kb " << sz / 1024 << " count " << count << " avg_elasped " << avg 
                << " throughput " << sz * count / 1024 / 1024 / avg * 1000 * 1000 
                << std::endl;

}
