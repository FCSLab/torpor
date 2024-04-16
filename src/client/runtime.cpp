#include "cuda_client.hpp"
#include <dlfcn.h>

using namespace client;

CUDAClient& cuda_client = CUDAClient::getInstance();

// cublas
cublasStatus_t cublasCreate(cublasHandle_t* handle) {
    return cuda_client.cublasCreate(handle);  
}

cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t streamId) {
    return cuda_client.cublasSetStream(handle, streamId);  
}

cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode) {
    return cuda_client.cublasSetMathMode(handle, mode);  
}

cublasStatus_t cublasGetMathMode(cublasHandle_t handle, cublasMath_t *mode) {
    return cuda_client.cublasGetMathMode(handle, mode);  
}

cublasStatus_t cublasSgemm(
        cublasHandle_t handle,
        cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k,
        const float           *alpha,
        const float           *A, int lda,
        const float           *B, int ldb,
        const float           *beta,
        float           *C, int ldc) {
    // std::cout << "cublasSgemm A " << std::to_string(reinterpret_cast<uint64_t>(A)) 
    //             << " B " << std::to_string(reinterpret_cast<uint64_t>(B)) 
    //             << " C " << std::to_string(reinterpret_cast<uint64_t>(C)) 
    //             << std::endl;
    return cuda_client.cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle,
        cublasOperation_t transa,
        cublasOperation_t transb,
        int m, int n, int k,
        const float           *alpha,
        const float           *A, int lda,
        long long int          strideA,
        const float           *B, int ldb,
        long long int          strideB,
        const float           *beta,
        float                 *C, int ldc,
        long long int          strideC,
        int batchCount) {
    return cuda_client.cublasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
}

// cudnn API
cudnnStatus_t cudnnCreate(cudnnHandle_t *handle){
    return cuda_client.cudnnCreate(handle);  
}

cudnnStatus_t cudnnSetStream(cudnnHandle_t handle, cudaStream_t stream) {
    return cuda_client.cudnnSetStream(handle, stream);  
}

cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t *tensorDesc) {
    return cuda_client.cudnnCreateTensorDescriptor(tensorDesc);      
}

cudnnStatus_t cudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t dataType, int nbDims, const int dimA[], const int strideA[]) {
    return cuda_client.cudnnSetTensorNdDescriptor(tensorDesc, dataType, nbDims, dimA, strideA);
}

cudnnStatus_t cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t *filterDesc) {
    return cuda_client.cudnnCreateFilterDescriptor(filterDesc);
}

cudnnStatus_t cudnnSetFilterNdDescriptor(cudnnFilterDescriptor_t filterDesc, cudnnDataType_t dataType, cudnnTensorFormat_t format, int nbDims, const int filterDimA[]) {
    return cuda_client.cudnnSetFilterNdDescriptor(filterDesc, dataType, format, nbDims, filterDimA);
}

cudnnStatus_t cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t *convDesc) {
    return cuda_client.cudnnCreateConvolutionDescriptor(convDesc);
}

cudnnStatus_t cudnnSetConvolutionNdDescriptor(
        cudnnConvolutionDescriptor_t    convDesc,
        int                             arrayLength,
        const int                       padA[],
        const int                       filterStrideA[],
        const int                       dilationA[],
        cudnnConvolutionMode_t          mode,
        cudnnDataType_t                 dataType) {
    return cuda_client.cudnnSetConvolutionNdDescriptor(convDesc, arrayLength, padA, filterStrideA, dilationA, mode, dataType);
}

cudnnStatus_t cudnnSetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc, int groupCount) {
    return cuda_client.cudnnSetConvolutionGroupCount(convDesc, groupCount);
}

cudnnStatus_t cudnnSetConvolutionMathType(cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t mathType) {
    return cuda_client.cudnnSetConvolutionMathType(convDesc, mathType);
}

cudnnStatus_t cudnnGetConvolutionForwardAlgorithm_v7(
        cudnnHandle_t                      handle,
        const cudnnTensorDescriptor_t       xDesc,
        const cudnnFilterDescriptor_t       wDesc,
        const cudnnConvolutionDescriptor_t  convDesc,
        const cudnnTensorDescriptor_t       yDesc,
        const int                           requestedAlgoCount,
        int                                *returnedAlgoCount,
        cudnnConvolutionFwdAlgoPerf_t      *perfResults) {
    return cuda_client.cudnnGetConvolutionForwardAlgorithm_v7(handle, xDesc, wDesc, convDesc, yDesc, requestedAlgoCount, returnedAlgoCount, perfResults);
}

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
    // std::cout << "cudnnConvolutionForward x " << std::to_string(reinterpret_cast<uint64_t>(x)) 
    //             << " w " << std::to_string(reinterpret_cast<uint64_t>(w)) 
    //             << " y " << std::to_string(reinterpret_cast<uint64_t>(y)) 
    //             << " space " << std::to_string(reinterpret_cast<uint64_t>(workSpace)) 
    //             << " algo " << std::to_string(static_cast<int>(algo)) 
    //             << " alpha " << std::to_string(*static_cast<const float*>(alpha)) 
    //             << " beta " << std::to_string(*static_cast<const float*>(beta)) 
    //             << std::endl;
    auto err = cuda_client.cudnnConvolutionForward(handle, alpha, xDesc, x, wDesc, w, convDesc, algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y);
    // cuda_client.cudaStreamSynchronize(0);
    // float xdst[2];
    // float wdst[2];
    // float ydst[2];
    // cuda_client.cudaMemcpy(xdst, x, 8, cudaMemcpyDeviceToHost);
    // cuda_client.cudaMemcpy(wdst, w, 8, cudaMemcpyDeviceToHost);
    // cuda_client.cudaMemcpy(ydst, y, 8, cudaMemcpyDeviceToHost);
    // std::cout << "Hook conv x [" << std::to_string(xdst[0]) << "," << std::to_string(xdst[1]) 
    //             << "] w[" << std::to_string(wdst[0]) << "," << std::to_string(wdst[1]) 
    //             << "] y[" << std::to_string(ydst[0]) << "," << std::to_string(ydst[1]) 
    //             << "]" << std::endl;
    return err;
}

cudnnStatus_t cudnnBatchNormalizationForwardInference(
      cudnnHandle_t                    handle,
      cudnnBatchNormMode_t             mode,
      const void                      *alpha,
      const void                      *beta,
      const cudnnTensorDescriptor_t    xDesc,
      const void                      *x,
      const cudnnTensorDescriptor_t    yDesc,
      void                            *y,
      const cudnnTensorDescriptor_t    bnScaleBiasMeanVarDesc,
      const void                      *bnScale,
      const void                      *bnBias,
      const void                      *estimatedMean,
      const void                      *estimatedVariance,
      double                           epsilon) {
    return cuda_client.cudnnBatchNormalizationForwardInference(handle, mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, estimatedMean, estimatedVariance, epsilon);    
}

cudnnStatus_t cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t convDesc) {
    return cuda_client.cudnnDestroyConvolutionDescriptor(convDesc);    
}

cudnnStatus_t cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t filterDesc) {
    return cuda_client.cudnnDestroyFilterDescriptor(filterDesc); 
}

cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc) {
    return cuda_client.cudnnDestroyTensorDescriptor(tensorDesc); 
}


// cuda runtime API
cudaError_t cudaDriverGetVersion(int* version) {
    return cuda_client.cudaDriverGetVersion(version);  
}

cudaError_t cudaRuntimeGetVersion(int* version) {
    return cuda_client.cudaRuntimeGetVersion(version);  
}

// error 
cudaError_t cudaGetLastError() {
    return cuda_client.cudaGetLastError();    
}

// device
cudaError_t cudaGetDeviceCount(int* count){
    return cuda_client.cudaGetDeviceCount(count);
}

cudaError_t cudaGetDevice(int* device){
    return cuda_client.cudaGetDevice(device);
}

cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device) {
    return cuda_client.cudaGetDeviceProperties(prop, device);
}

cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device) {
    return cuda_client.cudaDeviceGetAttribute(value, attr, device);
}

cudaError_t cudaDeviceSynchronize() {
    std::cout << "cudaDeviceSynchronize" << std::endl;
    return cuda_client.cudaDeviceSynchronize();
}

// memory
// cudaError_t cudaHostAlloc (void** pHost, size_t size, unsigned int flags) {
//     std::cout << "In cudaHostAlloc" << std::endl;
//     return static_cast<cudaError_t> (0);
// }

// cudaError_t cudaHostGetDevicePointer(void** pDevice, void* pHost, unsigned int flags) {
//     std::cout << "In cudaHostGetDevicePointer" << std::endl;
//     return static_cast<cudaError_t> (0);
// }

cudaError_t	cudaMalloc(void** devPtr, size_t size) {
    auto err =  cuda_client.cudaMalloc(devPtr, size);
    // std::cout << "cudaMalloc size " << std::to_string(size) << " ptr " << std::to_string(reinterpret_cast<uint64_t>(*devPtr)) << std::endl;
    return err;
}

cudaError_t	cudaFree(void* devPtr) {
    // std::cout << "cudaFree ptr " << std::to_string(reinterpret_cast<uint64_t>(devPtr)) << std::endl;
    return cuda_client.cudaFree(devPtr);
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    // std::cout << "cudaMemcpy kind " << std::to_string(kind) << std::endl;
    return cuda_client.cudaMemcpy(dst, src, count, kind);
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
    // std::cout << "cudaMemcpyAsync kind " << std::to_string(kind) 
    //             << " src " << std::to_string(reinterpret_cast<uint64_t>(src)) 
    //             << " count " << std::to_string(count) 
    //             << " dst " << std::to_string(reinterpret_cast<uint64_t>(dst)) 
    //             << " stream " << std::to_string(reinterpret_cast<uint64_t>(stream)) 
    //             << std::endl;
    return cuda_client.cudaMemcpyAsync(dst, src, count, kind, stream);
}

cudaError_t cudaMemsetAsync (void* devPtr, int value, size_t count, cudaStream_t stream) {
    // std::cout << "cudaMemsetAsync ptr " << std::to_string(reinterpret_cast<uint64_t>(devPtr))
    //             << " value " << std::to_string(value) 
    //             << " count " << std::to_string(count) 
    //             << " stream " << std::to_string(reinterpret_cast<uint64_t>(stream)) 
    //             << std::endl;
    return cuda_client.cudaMemsetAsync(devPtr, value, count, stream);  
}

cudaError_t cudaGetSymbolAddress (void** devPtr, const void* symbol) {
    return cuda_client.cudaGetSymbolAddress(devPtr, symbol);  
}

// stream
cudaError_t cudaStreamCreate(cudaStream_t* pstream) {
    return cuda_client.cudaStreamCreate(pstream);
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pstream, unsigned int flags) {
    return cuda_client.cudaStreamCreateWithFlags(pstream, flags);
}

cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pstream, unsigned int flags, int priority) {
    return cuda_client.cudaStreamCreateWithPriority(pstream, flags, priority);
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    // std::cout << "cudaStreamSynchronize" << std::endl;
    return cuda_client.cudaStreamSynchronize(stream);
}

cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus* pCaptureStatus) {
    return cuda_client.cudaStreamIsCapturing(stream, pCaptureStatus);
}

cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream, cudaStreamCaptureStatus* pCaptureStatus, unsigned long long* pId) {
    return cuda_client.cudaStreamGetCaptureInfo(stream, pCaptureStatus, pId);
}

cudaError_t cudaEventCreateWithFlags (cudaEvent_t* event, unsigned int flags) {
    return cuda_client.cudaEventCreateWithFlags(event, flags); 
} 

cudaError_t cudaEventQuery (cudaEvent_t event) {
    return cuda_client.cudaEventQuery(event); 
} 

cudaError_t cudaEventRecord (cudaEvent_t event, cudaStream_t stream) {
    return cuda_client.cudaEventRecord(event, stream); 
} 

std::map<const void *, std::string> &kernels() {
  static std::map<const void*, std::string> _kernels;
  return _kernels;
}
typedef void (*cudaRegisterFunction_t)(void **, const char *, char *,
                                         const char *, int, uint3 *,
                                         uint3 *, dim3 *, dim3 *, int *);
static cudaRegisterFunction_t realCudaRegisterFunction = NULL;

extern "C"
void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun, char *deviceFun,
                            const char *deviceName, int thread_limit, uint3 *tid,
                            uint3 *bid, dim3 *bDim, dim3 *gDim, int *wSize) {
    kernels()[hostFun] = std::string(deviceFun);
    if (realCudaRegisterFunction == NULL) {
        realCudaRegisterFunction = (cudaRegisterFunction_t)dlsym(RTLD_NEXT,"__cudaRegisterFunction");
    }
    realCudaRegisterFunction(fatCubinHandle, hostFun, deviceFun,
                            deviceName, thread_limit, tid,
                            bid, bDim, gDim, wSize);
}

cudaError_t cudaLaunchKernel (const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream) {
    // std::cout << "kernel count " << kernels().size() << std::endl;
    std::string fname("unknown");
    if (kernels().find(func) != kernels().end()) {
        fname = kernels()[func];
        // std::cout << "kernel name " << fname << std::endl;
        return cuda_client.cudaLaunchKernel(fname, gridDim, blockDim, args, sharedMem, stream); 
    }

    std::cout << "Unknown kernel" << std::endl;
    return static_cast<cudaError_t>(0);
}