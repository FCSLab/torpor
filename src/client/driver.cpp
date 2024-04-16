#include "cuda_client.hpp"

using namespace client;

CUDAClient& cuda_client = CUDAClient::getInstance();

// driver API (try workaround)
CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags, int* active) {
    return cuda_client.cuDevicePrimaryCtxGetState(dev, flags, active);     
}

CUresult cuInit (unsigned int flags){
    return cuda_client.cuInit(flags);     
}

// CUresult cuGetProcAddress (const char* symbol, void** pfn, int cudaVersion, cuuint64_t flags) {
//     return cuda_client.cuGetProcAddress(symbol, pfn, cudaVersion, flags);
// }

CUresult cuDriverGetVersion (int* driverVersion ) {
    return cuda_client.cuDriverGetVersion(driverVersion);
}