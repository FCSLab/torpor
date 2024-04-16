#include <stdio.h>
#include <iostream>
#include <unistd.h>
#include <dlfcn.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "kernel_lookup.hpp"

map<string, const void *> &lookup() {
  static map<string, const void *> _lookup;
  return _lookup;
}

map<string, KernelAccessLoc> &kernel_model_access_loc() {
  static map<string, KernelAccessLoc> _kernel_model_access_loc;
  return _kernel_model_access_loc;
}

typedef void (*cudaRegisterFunction_t)(void **, const char *, char *,
                                         const char *, int, uint3 *,
                                         uint3 *, dim3 *, dim3 *, int *);
static cudaRegisterFunction_t realCudaRegisterFunction = NULL;

extern "C"
void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun,
                            const char *deviceName, int thread_limit, uint3 *tid,
                            uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize) {
    lookup()[string(deviceFun)] = hostFun;
    if (realCudaRegisterFunction == NULL) {
        realCudaRegisterFunction = (cudaRegisterFunction_t)dlsym(RTLD_NEXT,"__cudaRegisterFunction");
    }

    realCudaRegisterFunction(fatCubinHandle, hostFun, deviceFun,
                            deviceName, thread_limit, tid,
                            bid, bDim, gDim, wSize);
}