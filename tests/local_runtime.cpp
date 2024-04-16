#include <stdio.h>
#include <iostream>
#include <unistd.h>
#include <dlfcn.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cublas_v2.h"
#include <cudnn.h>
#include <map>
#include <list>

cudnnStatus_t cudnnConvolutionForward(
        cudnnHandle_t                       handle,
        const void                         *alpha,
        const cudnnTensorDescriptor_t       xDesc,
        const void                         *x,
        const cudnnFilterDescriptor_t       wDesc,
        const void                         *w,
        const cudnnConvolutionDescriptor_t  convDesc,
        cudnnConvolutionFwdAlgo_t           algo,
        void                               *workSpace,
        size_t                              workSpaceSizeInBytes,
        const void                         *beta,
        const cudnnTensorDescriptor_t       yDesc,
        void                               *y) {
    std::cout << "cudnnConvolutionForward x " << std::to_string(reinterpret_cast<uint64_t>(x)) 
                << " w " << std::to_string(reinterpret_cast<uint64_t>(w)) 
                << " y " << std::to_string(reinterpret_cast<uint64_t>(y)) 
                << " space " << std::to_string(reinterpret_cast<uint64_t>(workSpace)) 
                << " algo " << std::to_string(static_cast<int>(algo)) 
                << " alpha " << std::to_string(*static_cast<const float*>(alpha)) 
                << " beta " << std::to_string(*static_cast<const float*>(beta)) 
                << std::endl;
    cudnnStatus_t (*lcudnnConvolutionForward) (cudnnHandle_t, const void*, const cudnnTensorDescriptor_t, const void*, const cudnnFilterDescriptor_t, const void*, const cudnnConvolutionDescriptor_t, cudnnConvolutionFwdAlgo_t, void*, size_t, const void*, const cudnnTensorDescriptor_t, void*) = 
        (cudnnStatus_t (*) (cudnnHandle_t, const void*, const cudnnTensorDescriptor_t, const void*, const cudnnFilterDescriptor_t, const void*, const cudnnConvolutionDescriptor_t, cudnnConvolutionFwdAlgo_t, void*, size_t, const void*, const cudnnTensorDescriptor_t, void*)) dlsym(RTLD_NEXT, "cudnnConvolutionForward");
    auto err = lcudnnConvolutionForward(handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y);

    float xdst[2];
    float wdst[2];
    float ydst[2];
    cudaMemcpy(xdst, x, 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(wdst, w, 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(ydst, y, 8, cudaMemcpyDeviceToHost);
    std::cout << "Hook conv x [" << std::to_string(xdst[0]) << "," << std::to_string(xdst[1]) 
                << "] w[" << std::to_string(wdst[0]) << "," << std::to_string(wdst[1]) 
                << "] y[" << std::to_string(ydst[0]) << "," << std::to_string(ydst[1]) 
                << "]" << std::endl;
    return err;
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
    std::cout << "cudaMemcpyAsync kind " << std::to_string(kind) 
                << " src " << std::to_string(reinterpret_cast<uint64_t>(src)) 
                << " count " << std::to_string(count) 
                << " dst " << std::to_string(reinterpret_cast<uint64_t>(dst)) 
                << " stream " << std::to_string(reinterpret_cast<uint64_t>(stream)) 
                << std::endl;
    cudaError_t (*lcudaMemcpyAsync) (void*, const void*, size_t, cudaMemcpyKind, cudaStream_t) = 
        (cudaError_t (*) (void*, const void*, size_t, cudaMemcpyKind, cudaStream_t)) dlsym(RTLD_NEXT, "cudaMemcpyAsync");
    return lcudaMemcpyAsync(dst, src, count, kind, stream);
}

cudaError_t	cudaMalloc(void** devPtr, size_t size) {
    cudaError_t (*lcudaMalloc) (void**, size_t) = (cudaError_t (*) (void**, size_t)) dlsym(RTLD_NEXT, "cudaMalloc");
    auto err = lcudaMalloc(devPtr, size);
    std::cout << "cudaMalloc size " << std::to_string(size) << " ptr " << std::to_string(reinterpret_cast<uint64_t>(*devPtr)) << std::endl;
    return err;
}

cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t *tensorDesc) {
    cudnnStatus_t (*lcudnnCreateTensorDescriptor) (cudnnTensorDescriptor_t*) = (cudnnStatus_t (*) (cudnnTensorDescriptor_t*)) dlsym(RTLD_NEXT, "cudnnCreateTensorDescriptor");
    auto err = lcudnnCreateTensorDescriptor(tensorDesc);
    std::cout << "cudnnCreateTensorDescriptor " << std::to_string(reinterpret_cast<uint64_t>((void*)tensorDesc)) << " * " << (uint64_t) (*tensorDesc) << std::endl;
    return err;
}

cudnnStatus_t cudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t dataType, int nbDims, const int dimA[], const int strideA[]) {
    cudnnStatus_t (*lcudnnSetTensorNdDescriptor) (cudnnTensorDescriptor_t, cudnnDataType_t, int, const int[], const int[]) = (cudnnStatus_t (*) (cudnnTensorDescriptor_t, cudnnDataType_t, int, const int[], const int[])) dlsym(RTLD_NEXT, "cudnnSetTensorNdDescriptor");
    auto err = lcudnnSetTensorNdDescriptor(tensorDesc, dataType, nbDims, dimA, strideA);
    std::cout << "cudnnSetTensorNdDescriptor " << std::to_string(reinterpret_cast<uint64_t>((void*)&tensorDesc)) << " * " << (uint64_t) (tensorDesc) << std::endl;
    return err;
}

// /* register */
// typedef struct {
//     dim3 gridDim;
//     dim3 blockDim;
//     int counter;
//     std::list<void *> args;
// } kernelInfo_t;

// kernelInfo_t &kernelInfo() {
//   static kernelInfo_t _kernelInfo;
//   return _kernelInfo;
// }

// std::map<const void *, std::string> &kernels() {
//   static std::map<const void*, std::string> _kernels;
//   return _kernels;
// }
// typedef void (*cudaRegisterFunction_t)(void **, const char *, char *,
//                                          const char *, int, uint3 *,
//                                          uint3 *, dim3 *, dim3 *, int *);
// static cudaRegisterFunction_t realCudaRegisterFunction = NULL;

// extern "C"
// void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun,
//                             const char *deviceName, int thread_limit, uint3 *tid,
//                             uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize) {
//     kernels()[hostFun] = std::string(deviceFun);
//     if (realCudaRegisterFunction == NULL) {
//         realCudaRegisterFunction = (cudaRegisterFunction_t)dlsym(RTLD_NEXT,"__cudaRegisterFunction");
//     }
//     // register_count += 1;
//     // std::cout << "__cudaRegisterFunction host func " << std::to_string((uint64_t)hostFun)
//     //             << " device func " << std::to_string((uint64_t)deviceFun)
//     //             << std::endl;
//     realCudaRegisterFunction(fatCubinHandle, hostFun, deviceFun,
//                             deviceName, thread_limit, tid,
//                             bid, bDim, gDim, wSize);
// }


// cudaError_t cudaLaunchKernel (const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream) {
//     std::string fname("unknown");
//     if (kernels().find(func) != kernels().end()) {
//         fname = kernels()[func];
//     }
//     std::cout << "cudaLaunchKernel func " << std::to_string((uint64_t)func)
//                 << " fname " << fname
//                 << " sharedMem " << sharedMem
//                 // << " arg " << arg_str
//                 << std::endl;
//     cudaError_t (*lcudaLaunchKernel) (const void*, dim3, dim3, void**, size_t, cudaStream_t) = (cudaError_t (*) (const void*, dim3, dim3, void**, size_t, cudaStream_t)) dlsym(RTLD_NEXT, "cudaLaunchKernel");
//     auto err = lcudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
//     return err;
// }


// CUresult cuModuleLoad(CUmodule* module, const char* fname) {
//     std::cout << "cuModuleLoad fname " << std::string(fname) << std::endl;

//     CUresult (*lcuModuleLoad) (CUmodule*, const char*) = (CUresult (*) (CUmodule*, const char*)) dlsym(RTLD_NEXT, "cuModuleLoad");
//     auto err = lcuModuleLoad(module, fname);
//     return err;
// }

// cudaError_t cudaGetSymbolAddress (void** devPtr, const void* symbol) {
//     std::cout << "cudaGetSymbolAddress symbol " << std::string(static_cast<const char*> (symbol)) << std::endl;
//     cudaError_t (*lcudaGetSymbolAddress) (void**, const void*) = (cudaError_t (*) (void**, const void*)) dlsym(RTLD_NEXT, "cudaGetSymbolAddress");
//     return lcudaGetSymbolAddress(devPtr, symbol);  
// }

// template < class T >
// cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags (int* numBlocks, T func, int blockSize, size_t dynamicSMemSize, unsigned int flags){
//     std::cout << "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags " << std::endl;
//     cudaError_t (*lcudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags) (int*, T, int, size_t, unsigned int) = (cudaError_t (*) (int*, T, int, size_t, unsigned int)) dlsym(RTLD_NEXT, "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags");
//     return lcudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, func, blockSize, dynamicSMemSize, flags);
// }

// g++ -I/usr/local/cuda/include -fPIC -shared -o libmylib.so mylib.cpp -ldl -L/usr/local/cuda/lib64 -lcudart
// LD_LIBRARY_PATH=/usr/local/cuda/lib64 LD_PRELOAD=./libmylib.so python cv.py