//#include <cudnn.h>
//#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include "proto/cudarpc.pb.h"
#include "/usr/local/cuda-11.4/targets/x86_64-linux/include/cublas_v2.h"
#include "/usr/local/cuda-11.4/targets/x86_64-linux/include/cudnn.h"
#include </usr/local/cuda-11.4/targets/x86_64-linux/include/cuda_runtime.h>

#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cout << "Error: " << __FILE__ << ":" << __LINE__ << ", " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

#define CHECK_CUDNN(call) { \
    const cudnnStatus_t status = call; \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::cout << "Error: " << __FILE__ << ":" << __LINE__ << ", " << cudnnGetErrorString(status) << std::endl; \
        exit(1); \
    } \
}


void cudaGetLastErrorService() {cudaGetLastError();}  // need to add cudacheck

int main() {
    int cnt = 2000; // number of apis

    for (int j=0; j<cnt; j++) {
        cudaGetLastErrorService();
    }
    cudaDeviceSynchronize();

    for (int i = 0; i <10; i++) {
        auto start = std::chrono::high_resolution_clock::now();        
        for (int j=0; j<cnt; j++) {
            cudaGetLastErrorService();
        }
        cudaDeviceSynchronize();
        auto stop = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Execution time: " << duration.count() << " ms" << std::endl;        
    }

    return 0;
}