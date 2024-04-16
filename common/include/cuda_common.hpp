#ifndef INCLUDE_CUDA_COMMON_HPP_
#define INCLUDE_CUDA_COMMON_HPP_

#include "types.hpp"
#include "cudarpc.pb.h"
#include "signal.pb.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cublas_v2.h"
#include <cudnn.h>

using signal::RequestType;
using signal::SignalRequest;
using signal::AckType;
using signal::SignalAck;

using cudarpc::QueryType;
using cudarpc::WapperQuery;
using cudarpc::QueryList;
using cudarpc::genericResponse;
using cudarpc::getVersionResponse;

using cudarpc::handleResponse;
using cudarpc::setStreamQuery;
using cudarpc::cublasSetMathModeQuery;
using cudarpc::cublasGetMathModeResponse;
using cudarpc::cublasSgemmQuery;
using cudarpc::cublasSgemmStridedBatchedQuery;

using cudarpc::cudnnCreateDesciptorResponse;
using cudarpc::cudnnSetTensorNdDescriptorQuery;
using cudarpc::cudnnSetFilterNdDescriptorQuery;
using cudarpc::cudnnSetConvolutionNdDescriptorQuery;
using cudarpc::cudnnSetConvolutionGroupCountQuery;
using cudarpc::cudnnSetConvolutionMathTypeQuery;
using cudarpc::cudnnGetConvolutionForwardAlgorithm_v7Query;
using cudarpc::cudnnGetConvolutionForwardAlgorithm_v7Response;
using cudarpc::cudnnConvolutionForwardQuery;
using cudarpc::cudnnBatchNormalizationForwardInferenceQuery;

using cudarpc::cuInitQuery;
using cudarpc::cuDevicePrimaryCtxGetStateQuery;
using cudarpc::cuDevicePrimaryCtxGetStateResponse;
using cudarpc::cuGetProcAddressQuery;
using cudarpc::cuGetProcAddressResponse;

using cudarpc::cudaGetDeviceResponse;
using cudarpc::cudaGetDeviceCountResponse;
using cudarpc::cudaGetDevicePropertiesQuery;
using cudarpc::cudaGetDevicePropertiesResponse;
using cudarpc::cudaDeviceGetAttributeQuery;
using cudarpc::cudaDeviceGetAttributeResponse;

using cudarpc::cudaMallocQuery;
using cudarpc::cudaMallocResponse;
using cudarpc::cudaFreeQuery;
using cudarpc::cudaMemcpyQuery;
using cudarpc::cudaMemcpyResponse;
using cudarpc::cudaMemcpyAsyncQuery;
using cudarpc::cudaMemcpyAsyncResponse;
using cudarpc::cudaMemsetAsyncQuery;
using cudarpc::cudaGetSymbolAddressQuery;
using cudarpc::cudaGetSymbolAddressResponse;

using cudarpc::AsyncResponse;
using cudarpc::cudaStreamCreateWithFlagsQuery;
using cudarpc::cudaStreamCreateWithPriorityQuery;
using cudarpc::cudaStreamCreateResponse;
using cudarpc::cudaStreamIsCapturingResponse;
using cudarpc::cudaStreamGetCaptureInfoResponse;

using cudarpc::cudaEventCreateWithFlagsQuery;
using cudarpc::cudaEventCreateWithFlagsResponse;
using cudarpc::cudaEventRecordQuery;

using cudarpc::cudaLaunchKernelQuery;

const set<QueryType> asyncQueryTypes = {
    QueryType::cuBLAS_cublasSetMathMode,
    QueryType::cuBLAS_cublasSetStream,
    QueryType::cuBLAS_cublasSgemm,
    QueryType::cuBLAS_cublasSgemmStridedBatched,

    QueryType::cudnnSetStream,
    QueryType::cudnnCreateFilterDescriptor,
    QueryType::cudnnSetFilterNdDescriptor,
    QueryType::cudnnDestroyFilterDescriptor,
    QueryType::cudnnCreateConvolutionDescriptor,
    QueryType::cudnnSetConvolutionGroupCount,
    QueryType::cudnnSetConvolutionNdDescriptor,
    QueryType::cudnnSetConvolutionMathType,
    QueryType::cudnnDestroyConvolutionDescriptor,
    QueryType::cudnnCreateTensorDescriptor,
    QueryType::cudnnSetTensorNdDescriptor,
    QueryType::cudnnDestroyTensorDescriptor,
    
    QueryType::cudnnConvolutionForward,
    QueryType::cudnnBatchNormalizationForwardInference,

    QueryType::cudaMemcpyAsync,
    QueryType::cudaMemsetAsync,
    QueryType::cudaGetLastError,
    QueryType::cudaLaunchKernel,
};

#define cudaCheck(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t err, const char *file, int line, bool abort=false) {
   if (err != cudaSuccess) {
      fprintf(stderr,"cuda assert: %s %s %d\n", cudaGetErrorString(err), file, line);
      if (abort) assert(0); 
   }
}

#define cudnnCheck(ans) { cudnnAssert((ans), __FILE__, __LINE__); }
inline void cudnnAssert(cudnnStatus_t err, const char *file, int line, bool abort=false) {
   if (err != CUDNN_STATUS_SUCCESS) {
      fprintf(stderr,"cudnn assert: %d %s %d\n", err, file, line);
      if (abort) assert(0); 
   }
}

#endif  // INCLUDE_CUDA_COMMON_HPP_