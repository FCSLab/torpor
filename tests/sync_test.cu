#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <chrono>


__global__ void add_kernel(int* A, int* B, int n, long long sleep_cycles) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;

    // __nanosleep(sleep_ns);
    long long start = clock64();
    long long cycles_elapsed;
    do { cycles_elapsed = clock64() - start; } 
    while (cycles_elapsed < sleep_cycles);

    if (k < n) 
        B[k] = B[k] + A[k];
}

// void CUDART_CB call_back(cudaStream_t stream, cudaError_t status, void *data){
//     std::cout << "recv call back " << *((int*) data) << std::endl;
//     readiness[*((int*) data)] = true;
// }


int main(int argc, char** argv) {
    int num = 1024 * 1024;
    int count = 1;
    int sleep_cycle_in_k = 100;
    int fuse_size = 1;
    if (argc > 1) {
        num = atoi(argv[1]);
    }

    if (argc > 2) {
        count = atoi(argv[2]);
    }

    if (argc > 3) {
        sleep_cycle_in_k = atoi(argv[3]);
    }

    if (argc > 4) {
        fuse_size = atoi(argv[4]);
    }
    size_t sz = num * sizeof(int);

    int blocksize = 512;
    int nblocks = num / blocksize;

    int sleep_cycles = sleep_cycle_in_k * 1000;

    cudaError_t err;
    int* host_ptrs[count];
    int* cuda_ptrs[count];
    int* re_ptr;
    
    cudaEvent_t event;
    cudaEventCreate(&event);

    for (int i = 0; i < count; i++){
        err = cudaMallocHost(&host_ptrs[i], sz);
        for (int j = 0; j < num; j++) {
            host_ptrs[i][j] = 1;
        }
        err = cudaMalloc(&cuda_ptrs[i], sz);
    }

    err = cudaMalloc(&re_ptr, sz);

    cudaStream_t stream_1, stream_2;
    cudaStreamCreate(&stream_1);
    cudaStreamCreate(&stream_2);

    // warmup 
    add_kernel<<<nblocks, blocksize, 0, stream_2>>>(cuda_ptrs[0], re_ptr, num, sleep_cycles);
    cudaDeviceSynchronize();

    auto start_t = std::chrono::system_clock::now();

    for (int i = 0; i < count; i++){
        err = cudaMemcpyAsync(cuda_ptrs[i], host_ptrs[i], sz, cudaMemcpyKind::cudaMemcpyHostToDevice, stream_1);

        if ((i + 1) % fuse_size == 0) {
            cudaEventRecord(event, stream_1);
            cudaEventSynchronize(event);

            for (int j = i + 1 - fuse_size; j <= i; j++) {
                add_kernel<<<nblocks, blocksize, 0, stream_2>>>(cuda_ptrs[j], re_ptr, num, sleep_cycles);
            }
        }
    }

    // for (int i = 0; i < count; i++){
    //     add_kernel<<<nblocks, blocksize, 0, stream_2>>>(cuda_ptrs[i], re_ptr, num, sleep_cycles);
    // }

    cudaDeviceSynchronize();

    auto end_t = std::chrono::system_clock::now();
    auto elasped = std::chrono::duration_cast<std::chrono::microseconds> (end_t - start_t).count();

    int re[2];
    cudaMemcpy(re, re_ptr, sizeof(re), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    std::cout << "Result [" << std::to_string(re[0]) << ", " << std::to_string(re[1]) << "]" << std::endl;

    std::cout << "Sync num " << num << " sz " << sz << " count " << count << " elasped " << std::to_string(elasped) << std::endl;
}
