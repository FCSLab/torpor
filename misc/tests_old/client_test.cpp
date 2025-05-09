#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <chrono>

int main(){

    int count = 0;
    auto start_t = std::chrono::system_clock::now();
    auto err = cudaGetDeviceCount(&count);
    auto elasped = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start_t).count();
    std::cout << "cudaGetDeviceCount " << count << ". elasped " << elasped << " err " << err <<  std::endl;

    if (count > 0) {
        cudaDeviceProp prop;
        start_t = std::chrono::system_clock::now();
        err = cudaGetDeviceProperties(&prop, 0);   
        elasped = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start_t).count();
        std::cout << "cudaGetDeviceProperties " << ". elasped " << elasped << " err " << err << std::endl;

        std::stringstream ss;
        ss << std::endl;
        ss << "Device id:                     " << 0 << std::endl;
        ss << "Major revision number:         " << prop.major << std::endl;
        ss << "Minor revision number:         " << prop.minor << std::endl;
        ss << "Name:                          " << prop.name << std::endl;
        ss << "Total global memory:           " << prop.totalGlobalMem << std::endl;
        ss << "Total shared memory per block: " << prop.sharedMemPerBlock
            << std::endl;
        ss << "Total registers per block:     " << prop.regsPerBlock << std::endl;
        ss << "Warp size:                     " << prop.warpSize << std::endl;
        ss << "Maximum memory pitch:          " << prop.memPitch << std::endl;
        ss << "Maximum threads per block:     " << prop.maxThreadsPerBlock
            << std::endl;
        ss << "Maximum dimension of block:    "
            << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", "
            << prop.maxThreadsDim[2] << std::endl;
        ss << "Maximum dimension of grid:     "
            << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", "
            << prop.maxGridSize[2] << std::endl;
        ss << "Clock rate:                    " << prop.clockRate << std::endl;
        ss << "Total constant memory:         " << prop.totalConstMem << std::endl;
        ss << "Texture alignment:             " << prop.textureAlignment << std::endl;
        ss << "Concurrent copy and execution: "
            << (prop.deviceOverlap ? "Yes" : "No") << std::endl;
        ss << "Number of multiprocessors:     " << prop.multiProcessorCount
            << std::endl;
        ss << "Kernel execution timeout:      "
            << (prop.kernelExecTimeoutEnabled ? "Yes" : "No") << std::endl;
        std::cout << ss.str();

        start_t = std::chrono::system_clock::now();
        err = cudaDeviceSynchronize();   
        elasped = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start_t).count();
        std::cout << "cudaDeviceSynchronize. elasped " << elasped << " err " << err << std::endl;

        start_t = std::chrono::system_clock::now();
        cudaStream_t cuda_stream;
        err = cudaStreamCreateWithPriority(&cuda_stream, cudaStreamNonBlocking, -1);
        elasped = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start_t).count();
        std::cout << "cudaStreamCreateWithPriority. elasped " << elasped << " err " << err << " stream "<< cuda_stream << std::endl;
        
        start_t = std::chrono::system_clock::now();
        err = cudaStreamSynchronize(cuda_stream);
        elasped = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start_t).count();
        std::cout << "cudaStreamSynchronize. elasped " << elasped << " err " << err << " stream "<< cuda_stream << std::endl;

        start_t = std::chrono::system_clock::now();
        cudaStreamCaptureStatus status;
        err = cudaStreamIsCapturing(cuda_stream, &status);
        elasped = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start_t).count();
        std::cout << "cudaStreamIsCapturing " << status << ". elasped " << elasped << " err " << err << " stream "<< cuda_stream << std::endl;


        size_t sz = 1024 * 1024 * 10;
        // size_t sz = 100;
        float* ptr_1 = nullptr;
        float* ptr_2 = nullptr;
        err = cudaMalloc(&ptr_1, sz);
        err = cudaMalloc(&ptr_2, sz);
        auto ptr_int = reinterpret_cast<uint64_t>(ptr_1);
        std::cout << "cudaMalloc " << ptr_1 << " toInt " << ptr_int << " toPtr " << reinterpret_cast<void*>(ptr_int) << std::endl;

        cudaStream_t another_cuda_stream;
        err = cudaStreamCreateWithPriority(&another_cuda_stream, cudaStreamNonBlocking, -1);
        std::cout << "another stream err " << err << " stream "<< another_cuda_stream << std::endl;

        float* src_1 = new float[sz]();
        float* src_2 = new float[sz]();
        src_1[0] = 7;
        src_2[0] = 8;
        // float src_1[sz] = {7};
        // float src_2[sz] = {8};
        std::cout << "prepared src" << std::endl;
        start_t = std::chrono::system_clock::now();
        err = cudaMemcpyAsync(ptr_1, src_1, sz, cudaMemcpyKind::cudaMemcpyHostToDevice, cuda_stream);
        std::cout << "issue first" << std::endl;
        auto start2_t = std::chrono::system_clock::now();
        err = cudaMemcpyAsync(ptr_2, src_2, sz, cudaMemcpyKind::cudaMemcpyHostToDevice, another_cuda_stream);
        auto issue_t = std::chrono::system_clock::now();

        err = cudaStreamSynchronize(cuda_stream);
        auto sync_t = std::chrono::system_clock::now();
        err = cudaStreamSynchronize(another_cuda_stream);
        auto sync2_t = std::chrono::system_clock::now();

        std::cout << "cudaMemcpyAsync issue first " 
                    << std::chrono::duration_cast<std::chrono::microseconds>(start2_t - start_t).count() 
                    << " issue second " << std::chrono::duration_cast<std::chrono::microseconds>(issue_t - start2_t).count() 
                    << " sync first " << std::chrono::duration_cast<std::chrono::microseconds>(sync_t - issue_t).count() 
                    << " sync second " << std::chrono::duration_cast<std::chrono::microseconds>(sync2_t - sync_t).count() 
                    << std::endl;

        auto res = static_cast<float*>(malloc(sz));
        err = cudaMemcpyAsync(res, ptr_1, sz, cudaMemcpyKind::cudaMemcpyDeviceToHost, cuda_stream);
        err = cudaStreamSynchronize(cuda_stream);
        std::cout << "cudaMemcpy " << src_1[0] << " back " << res[0] << std::endl;

        err = cudaMemcpyAsync(res, ptr_2, sz, cudaMemcpyKind::cudaMemcpyDeviceToHost, another_cuda_stream);
        err = cudaStreamSynchronize(another_cuda_stream);
        std::cout << "cudaMemcpy " << src_2[0] << " back " << res[0] << std::endl;
    }
}
