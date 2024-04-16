#include <stdio.h>
#include <iostream>
#include <unistd.h>
#include <dlfcn.h>
#include <cuda.h>

CUresult cuGetProcAddress (const char* symbol, void** pfn, int cudaVersion, cuuint64_t flags) {
    std::cout << "[driver] cuGetProcAddress hook " << symbol << std::endl;
    CUresult (*lcuGetProcAddress) (const char*, void**, int, cuuint64_t) = (CUresult (*) (const char*, void**, int, cuuint64_t)) dlsym(RTLD_NEXT, "cuGetProcAddress");
    return lcuGetProcAddress(symbol, pfn, cudaVersion, flags);
}
