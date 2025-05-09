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
#include "api_wapper.hpp"

kernelInfo_t &kernelInfo() {
  static kernelInfo_t _kernelInfo;
  return _kernelInfo;
}

std::map<const void *, std::string> &kernels() {
  static std::map<const void*, std::string> _kernels;
  return _kernels;
}

std::map<std::string, const void *> &lookup() {
  static std::map<std::string, const void *> _lookup;
  return _lookup;
}

// typedef cudaError_t (*cudaConfigureCall_t)(dim3,dim3,size_t,cudaStream_t);
// static cudaConfigureCall_t realCudaConfigureCall = NULL;

// extern "C"
// cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem=0, cudaStream_t stream=0) {
//     kernelInfo().gridDim = gridDim;
//     kernelInfo().blockDim = blockDim;
//     kernelInfo().counter++;
//     if (realCudaConfigureCall == NULL)
//       realCudaConfigureCall = (cudaConfigureCall_t)dlsym(RTLD_NEXT,"cudaConfigureCall");
//     return realCudaConfigureCall(gridDim,blockDim,sharedMem,stream);
// }


typedef void (*cudaRegisterFunction_t)(void **, const char *, char *,
                                         const char *, int, uint3 *,
                                         uint3 *, dim3 *, dim3 *, int *);
static cudaRegisterFunction_t realCudaRegisterFunction = NULL;

extern "C"
void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun,
                            const char *deviceName, int thread_limit, uint3 *tid,
                            uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize) {
    kernels()[hostFun] = std::string(deviceFun);
    lookup()[std::string(deviceFun)] = hostFun;
    if (realCudaRegisterFunction == NULL) {
        realCudaRegisterFunction = (cudaRegisterFunction_t)dlsym(RTLD_NEXT,"__cudaRegisterFunction");
    }
    // register_count += 1;
    // std::cout << "__cudaRegisterFunction host func " << std::to_string((uint64_t)hostFun)
    //             << " device func " << std::to_string((uint64_t)deviceFun)
    //             << std::endl;
    realCudaRegisterFunction(fatCubinHandle, hostFun, deviceFun,
                            deviceName, thread_limit, tid,
                            bid, bDim, gDim, wSize);
}

// typedef cudaError_t (*cudaSetupArgument_t)(const void *, size_t, size_t);
// static cudaSetupArgument_t realCudaSetupArgument = NULL;

// extern "C"
// cudaError_t cudaSetupArgument(const void *arg, size_t size, size_t offset) {
//   kernelInfo().args.push_back(std::make_pair(const_cast<void *>(arg), size));
//   if (realCudaSetupArgument == NULL) {
//     realCudaSetupArgument = (cudaSetupArgument_t)dlsym(RTLD_NEXT,"cudaSetupArgument");
//   }
//   return realCudaSetupArgument(arg, size, offset);
// }

cudaError_t cudaLaunchKernel (const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream) {
    std::string fname("unknown");
    if (kernels().find(func) != kernels().end()) {
        fname = kernels()[func];
    }
    std::cout << "cudaLaunchKernel func " << std::to_string((uint64_t)func)
                << " fname " << fname
                // << " kernelInfoCounter " << kernelInfo().counter
                << " sharedMem " << sharedMem
                // << " kernelInfoArgs " << kernelInfo().args.size()
                << std::endl;
    kernelInfo().counter--; kernelInfo().args.clear();

    cudaError_t (*lcudaLaunchKernel) (const void*, dim3, dim3, void**, size_t, cudaStream_t) = (cudaError_t (*) (const void*, dim3, dim3, void**, size_t, cudaStream_t)) dlsym(RTLD_NEXT, "cudaLaunchKernel");
    auto err = lcudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
    return err;
}


CUresult cuModuleLoad(CUmodule* module, const char* fname) {
    std::cout << "cuModuleLoad fname " << std::string(fname) << std::endl;

    CUresult (*lcuModuleLoad) (CUmodule*, const char*) = (CUresult (*) (CUmodule*, const char*)) dlsym(RTLD_NEXT, "cuModuleLoad");
    auto err = lcuModuleLoad(module, fname);
    return err;
}

CUresult cuModuleLoadData ( CUmodule* module, const void* image) {
    std::cout << "cuModuleLoadData image " << std::endl;

    CUresult (*lcuModuleLoadData) (CUmodule*, const void*) = (CUresult (*) (CUmodule*, const void*)) dlsym(RTLD_NEXT, "cuModuleLoadData");
    auto err = lcuModuleLoadData(module, image);
    return err;
}

CUresult cuModuleGetFunction(CUfunction* hfun, CUmodule hmod, const char* name) {
    std::cout << "cuModuleGetFunction fname " << std::string(name) << std::endl;
    CUresult (*lcuModuleGetFunction) (CUfunction*, CUmodule, const char*) = (CUresult (*) (CUfunction*, CUmodule, const char*)) dlsym(RTLD_NEXT, "cuModuleGetFunction");
    auto err = lcuModuleGetFunction(hfun, hmod, name);
    return err;
}
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

// g++ -I/usr/local/cuda/include -fPIC -shared -o libmylib.so api_wapper.cpp -ldl -L/usr/local/cuda/lib64 -lcudart
// LD_LIBRARY_PATH=/usr/local/cuda/lib64 LD_PRELOAD=./libmylib.so python cv.py