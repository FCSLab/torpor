#ifndef INCLUDE_CUDA_CLIENT_HPP_
#define  INCLUDE_CUDA_CLIENT_HPP_

#include "cuda_common.hpp"
#include "async_sender.hpp"
#include <sstream>
#include <fstream>

namespace client{

class CUDAClient {
public:
    static CUDAClient& getInstance() {
        static CUDAClient client;
        return client;
    }
    
    /**
    * cublas API
    */
    cublasStatus_t cublasCreate(cublasHandle_t* handle) {
        WapperQuery req;
        req.set_type(QueryType::cuBLAS_cublasCreate); 

        auto resp = sender_.send_and_recv<handleResponse>(req);

        memcpy(handle, &resp.handle()[0], resp.handle().length());
        return static_cast<cublasStatus_t>(resp.error());
    }

    cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t stream) {
        WapperQuery req;
        req.set_type(QueryType::cuBLAS_cublasSetStream); 
        setStreamQuery query;
        query.set_handle(string(reinterpret_cast<char*> (&handle), sizeof(handle)));
        query.set_stream(string(reinterpret_cast<char*> (&stream), sizeof(stream)));
        string query_serialized;
        query.SerializeToString(&query_serialized); 
        req.set_query(query_serialized);

        // auto resp = send_and_recv<genericResponse>(req);
        // return static_cast<cublasStatus_t>(resp.error());

        sender_.send_async(req);
        return static_cast<cublasStatus_t>(0);
    }

    cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode) {
        WapperQuery req;
        req.set_type(QueryType::cuBLAS_cublasSetMathMode); 
        cublasSetMathModeQuery query;
        query.set_handle(string(reinterpret_cast<char*> (&handle), sizeof(handle)));
        query.set_mode(static_cast<int> (mode));
        string query_serialized;
        query.SerializeToString(&query_serialized); 
        req.set_query(query_serialized);

        // auto resp = send_and_recv<genericResponse>(req);
        // return static_cast<cublasStatus_t>(resp.error());
        sender_.send_async(req);
        return static_cast<cublasStatus_t>(0);
    }

    cublasStatus_t cublasGetMathMode(cublasHandle_t handle, cublasMath_t *mode) {
        WapperQuery req;
        req.set_type(QueryType::cuBLAS_cublasGetMathMode); 
        req.set_query(string(reinterpret_cast<char*> (&handle), sizeof(handle)));

        auto resp = sender_.send_and_recv<cublasGetMathModeResponse>(req);
        *mode = static_cast<cublasMath_t> (resp.mode());
        return static_cast<cublasStatus_t>(resp.error());
    }

    cublasStatus_t cublasSgemm(cublasHandle_t handle,
                            cublasOperation_t transa, cublasOperation_t transb,
                            int m, int n, int k,
                            const float           *alpha,
                            const float           *A, int lda,
                            const float           *B, int ldb,
                            const float           *beta,
                            float           *C, int ldc) {
        WapperQuery req;
        req.set_type(QueryType::cuBLAS_cublasSgemm); 
        
        cublasSgemmQuery query;
        query.set_handle(string(reinterpret_cast<char*> (&handle), sizeof(handle)));
        query.set_transa(static_cast<int> (transa));
        query.set_transb(static_cast<int> (transb));
        query.set_m(m);
        query.set_n(n);
        query.set_k(k);
        query.set_alpha(*alpha);
        query.set_matrix_a(reinterpret_cast<uint64_t>(A));
        query.set_lda(lda);
        query.set_matrix_b(reinterpret_cast<uint64_t>(B));
        query.set_ldb(ldb);
        query.set_beta(*beta);
        query.set_matrix_c(reinterpret_cast<uint64_t>(C));
        query.set_ldc(ldc);

        string query_serialized;
        query.SerializeToString(&query_serialized); 
        req.set_query(query_serialized);

        sender_.send_async(req);
        return static_cast<cublasStatus_t>(0);
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

        WapperQuery req;
        req.set_type(QueryType::cuBLAS_cublasSgemmStridedBatched);    

        cublasSgemmStridedBatchedQuery query;
        query.set_handle(string(reinterpret_cast<char*> (&handle), sizeof(handle)));
        query.set_transa(static_cast<int> (transa));
        query.set_transb(static_cast<int> (transb));
        query.set_m(m);
        query.set_n(n);
        query.set_k(k);
        query.set_alpha(*alpha);
        query.set_matrix_a(reinterpret_cast<uint64_t>(A));
        query.set_lda(lda);
        query.set_matrix_b(reinterpret_cast<uint64_t>(B));
        query.set_ldb(ldb);
        query.set_beta(*beta);
        query.set_matrix_c(reinterpret_cast<uint64_t>(C));
        query.set_ldc(ldc);
        query.set_stride_a(strideA);
        query.set_stride_b(strideB);
        query.set_stride_c(strideC);
        query.set_count(batchCount);

        string query_serialized;
        query.SerializeToString(&query_serialized); 
        req.set_query(query_serialized);

        sender_.send_async(req);
        return static_cast<cublasStatus_t>(0);
    }


    /**
    * cudnn API
    */
    cudnnStatus_t cudnnCreate(cudnnHandle_t *handle){
        WapperQuery req;
        req.set_type(QueryType::cudnnCreate); 

        auto resp = sender_.send_and_recv<handleResponse>(req);
        memcpy(handle, &resp.handle()[0], resp.handle().length());

        return static_cast<cudnnStatus_t>(resp.error());
    }

    cudnnStatus_t cudnnSetStream(cudnnHandle_t handle, cudaStream_t stream) {
        WapperQuery req;
        req.set_type(QueryType::cudnnSetStream); 
        setStreamQuery query;
        query.set_handle(string(reinterpret_cast<char*> (&handle), sizeof(handle)));
        query.set_stream(string(reinterpret_cast<char*> (&stream), sizeof(stream)));
        string query_serialized;
        query.SerializeToString(&query_serialized); 
        req.set_query(query_serialized);

        sender_.send_async(req);
        return static_cast<cudnnStatus_t>(0);
    }

    cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t *tensorDesc) {
        WapperQuery req;
        req.set_type(QueryType::cudnnCreateTensorDescriptor); 

        *tensorDesc = (cudnnTensorDescriptor_t) get_object_id();
        req.set_query(std::to_string((uint64_t) (*tensorDesc)));
        // std::cout << "cudnnCreateTensorDescriptor: " << (uint64_t) (*tensorDesc) << std::endl;

        // auto resp = send_and_recv<cudnnCreateDesciptorResponse>(req);
        // memcpy(tensorDesc, &resp.cudnn_descriptor()[0], resp.cudnn_descriptor().length());
        sender_.send_async(req);
        return static_cast<cudnnStatus_t>(0);  
    }

    cudnnStatus_t cudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t tensorDesc, cudnnDataType_t dataType, int nbDims, const int dimA[], const int strideA[]) {
        WapperQuery req;
        req.set_type(QueryType::cudnnSetTensorNdDescriptor); 

        cudnnSetTensorNdDescriptorQuery query;
        // std::cout << "cudnnSetTensorNdDescriptor: " << (uint64_t) (tensorDesc) << std::endl;
        query.set_cudnn_descriptor(std::to_string((uint64_t) tensorDesc));
        query.set_type(static_cast<int> (dataType));
        query.set_dims(nbDims);

        for (int i = 0; i < nbDims; i++) {
            query.add_dim_a(dimA[i]);
            query.add_stride_a(strideA[i]);
        }

        string query_serialized;
        query.SerializeToString(&query_serialized); 
        req.set_query(query_serialized);  

        // auto resp = send_and_recv<genericResponse>(req);
        // return static_cast<cudnnStatus_t>(resp.error());
        sender_.send_async(req);
        return static_cast<cudnnStatus_t>(0);  
    }

    cudnnStatus_t cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t *filterDesc) {
        WapperQuery req;
        req.set_type(QueryType::cudnnCreateFilterDescriptor); 

        *filterDesc = (cudnnFilterDescriptor_t) get_object_id();
        req.set_query(std::to_string((uint64_t) *filterDesc));

        // auto resp = send_and_recv<cudnnCreateDesciptorResponse>(req);
        // memcpy(filterDesc, &resp.cudnn_descriptor()[0], resp.cudnn_descriptor().length());
        // return static_cast<cudnnStatus_t>(resp.error()); 
        sender_.send_async(req);
        return static_cast<cudnnStatus_t>(0);  
    }

    cudnnStatus_t cudnnSetFilterNdDescriptor(cudnnFilterDescriptor_t filterDesc, cudnnDataType_t dataType, cudnnTensorFormat_t format, int nbDims, const int filterDimA[]) {
        WapperQuery req;
        req.set_type(QueryType::cudnnSetFilterNdDescriptor); 

        cudnnSetFilterNdDescriptorQuery query;
        query.set_cudnn_descriptor(std::to_string((uint64_t) filterDesc));
        query.set_type(static_cast<int> (dataType));
        query.set_format(static_cast<int> (format));
        query.set_dims(nbDims);

        for (int i = 0; i < nbDims; i++) {
            query.add_dim_a(filterDimA[i]);
        }

        string query_serialized;
        query.SerializeToString(&query_serialized); 
        req.set_query(query_serialized);  

        sender_.send_async(req);
        return static_cast<cudnnStatus_t>(0);  
    }

    cudnnStatus_t cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t *convDesc) {
        WapperQuery req;
        req.set_type(QueryType::cudnnCreateConvolutionDescriptor); 

        *convDesc = (cudnnConvolutionDescriptor_t) get_object_id();
        req.set_query(std::to_string((uint64_t) *convDesc));

        // auto resp = send_and_recv<cudnnCreateDesciptorResponse>(req);
        // memcpy(convDesc, &resp.cudnn_descriptor()[0], resp.cudnn_descriptor().length());

        sender_.send_async(req);
        return static_cast<cudnnStatus_t>(0); 
    }

    cudnnStatus_t cudnnSetConvolutionNdDescriptor(
            cudnnConvolutionDescriptor_t    convDesc,
            int                             arrayLength,
            const int                       padA[],
            const int                       filterStrideA[],
            const int                       dilationA[],
            cudnnConvolutionMode_t          mode,
            cudnnDataType_t                 dataType) {
        WapperQuery req;
        req.set_type(QueryType::cudnnSetConvolutionNdDescriptor); 

        cudnnSetConvolutionNdDescriptorQuery query;
        query.set_cudnn_descriptor(std::to_string((uint64_t) convDesc));
        query.set_type(static_cast<int> (dataType));
        query.set_mode(static_cast<int> (mode));
        query.set_length(arrayLength);

        for (int i = 0; i < arrayLength; i++) {
            query.add_pad_a(padA[i]);
            query.add_stride_a(filterStrideA[i]);
            query.add_dilation_a(dilationA[i]);
        }

        string query_serialized;
        query.SerializeToString(&query_serialized); 
        req.set_query(query_serialized);  
        sender_.send_async(req);
        return static_cast<cudnnStatus_t>(0); 
    }

    cudnnStatus_t cudnnSetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc, int groupCount) {
        WapperQuery req;
        req.set_type(QueryType::cudnnSetConvolutionGroupCount); 

        cudnnSetConvolutionGroupCountQuery query;
        query.set_cudnn_descriptor(std::to_string((uint64_t) convDesc));
        query.set_count(groupCount);

        string query_serialized;
        query.SerializeToString(&query_serialized); 
        req.set_query(query_serialized);  

        sender_.send_async(req);
        return static_cast<cudnnStatus_t>(0); 
    }

    cudnnStatus_t cudnnSetConvolutionMathType(cudnnConvolutionDescriptor_t convDesc, cudnnMathType_t mathType) {
        WapperQuery req;
        req.set_type(QueryType::cudnnSetConvolutionMathType); 

        cudnnSetConvolutionMathTypeQuery query;
        query.set_cudnn_descriptor(std::to_string((uint64_t) convDesc));
        query.set_type(static_cast<int> (mathType));

        string query_serialized;
        query.SerializeToString(&query_serialized); 
        req.set_query(query_serialized);  
        
        sender_.send_async(req);
        return static_cast<cudnnStatus_t>(0);   
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
        WapperQuery req;
        req.set_type(QueryType::cudnnGetConvolutionForwardAlgorithm_v7); 

        cudnnGetConvolutionForwardAlgorithm_v7Query query;
        query.set_handle(string(reinterpret_cast<char*>(&handle), sizeof(handle)));
        query.set_x_desc(std::to_string((uint64_t) xDesc));
        query.set_w_desc(std::to_string((uint64_t) wDesc));
        query.set_conv_desc(std::to_string((uint64_t) convDesc));
        query.set_y_desc(std::to_string((uint64_t) yDesc));
        query.set_count(requestedAlgoCount);

        string query_serialized;
        query.SerializeToString(&query_serialized); 
        req.set_query(query_serialized);  

        auto resp = sender_.send_and_recv<cudnnGetConvolutionForwardAlgorithm_v7Response>(req);
        *returnedAlgoCount = resp.count();
        memcpy(perfResults, &resp.results()[0], resp.results().length());
        return static_cast<cudnnStatus_t>(resp.error());    
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
        WapperQuery req;
        req.set_type(QueryType::cudnnConvolutionForward);

        cudnnConvolutionForwardQuery query;
        query.set_handle(string(reinterpret_cast<char*>(&handle), sizeof(handle)));
        query.set_x_desc(std::to_string((uint64_t) xDesc));
        query.set_w_desc(std::to_string((uint64_t) wDesc));
        query.set_conv_desc(std::to_string((uint64_t) convDesc));
        query.set_y_desc(std::to_string((uint64_t) yDesc));
        query.set_algo(static_cast<int> (algo));

        query.set_alpha(*static_cast<const float*>(alpha));
        query.set_beta(*static_cast<const float*>(beta));
        query.set_x(reinterpret_cast<uint64_t> (x));
        query.set_w(reinterpret_cast<uint64_t> (w));
        query.set_y(reinterpret_cast<uint64_t> (y));
        query.set_workspace(reinterpret_cast<uint64_t> (workSpace));
        query.set_workspace_size(workSpaceSizeInBytes);

        string query_serialized;
        query.SerializeToString(&query_serialized); 
        req.set_query(query_serialized);  
        
        sender_.send_async(req);
        return static_cast<cudnnStatus_t>(0);
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
        WapperQuery req;
        req.set_type(QueryType::cudnnBatchNormalizationForwardInference);

        cudnnBatchNormalizationForwardInferenceQuery query;
        query.set_handle(string(reinterpret_cast<char*>(&handle), sizeof(handle)));
        query.set_x_desc(std::to_string((uint64_t) xDesc));
        query.set_y_desc(std::to_string((uint64_t) yDesc));
        query.set_bn_desc(std::to_string((uint64_t) bnScaleBiasMeanVarDesc));

        query.set_epsilon(epsilon);
        query.set_mode(static_cast<int> (mode));
        query.set_alpha(*static_cast<const float*>(alpha));
        query.set_beta(*static_cast<const float*>(beta));

        query.set_x(reinterpret_cast<uint64_t> (x));
        query.set_y(reinterpret_cast<uint64_t> (y));
        query.set_bn_scale(reinterpret_cast<uint64_t> (bnScale));
        query.set_bn_bias(reinterpret_cast<uint64_t> (bnBias));
        query.set_es_mean(reinterpret_cast<uint64_t> (estimatedMean));
        query.set_es_var(reinterpret_cast<uint64_t> (estimatedVariance));

        string query_serialized;
        query.SerializeToString(&query_serialized); 
        req.set_query(query_serialized);  
        
        sender_.send_async(req);
        return static_cast<cudnnStatus_t>(0);
    }

    cudnnStatus_t cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t convDesc) {
        WapperQuery req;
        req.set_type(QueryType::cudnnDestroyConvolutionDescriptor);
        req.set_query(std::to_string((uint64_t) convDesc));

        sender_.send_async(req);
        return static_cast<cudnnStatus_t>(0);
    }

    cudnnStatus_t cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t filterDesc) {
        WapperQuery req;
        req.set_type(QueryType::cudnnDestroyFilterDescriptor);
        req.set_query(std::to_string((uint64_t) filterDesc));

        sender_.send_async(req);
        return static_cast<cudnnStatus_t>(0);  
    }

    cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc) {
        WapperQuery req;
        req.set_type(QueryType::cudnnDestroyTensorDescriptor);
        req.set_query(std::to_string((uint64_t) tensorDesc));

        sender_.send_async(req);
        return static_cast<cudnnStatus_t>(0);  
    }

    /**
    * CUDA driver API
    */
    CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int* flags, int* active) {
        WapperQuery req;
        req.set_type(QueryType::cuDevicePrimaryCtxGetState);

        cuDevicePrimaryCtxGetStateQuery query;
        query.set_device(static_cast<int64_t>(dev));
        string query_serialized;
        query.SerializeToString(&query_serialized); 
        req.set_query(query_serialized);

        auto resp = sender_.send_and_recv<cuDevicePrimaryCtxGetStateResponse>(req);
        *flags = resp.flags();
        *active = resp.active();
        return static_cast<CUresult>(resp.error());
    }

    CUresult cuInit (unsigned int flags){
        WapperQuery req;
        req.set_type(QueryType::cuInit);

        cuInitQuery query;
        query.set_flags(flags);
        string query_serialized;
        query.SerializeToString(&query_serialized); 
        req.set_query(query_serialized);

        auto resp = sender_.send_and_recv<genericResponse>(req);
        return static_cast<CUresult>(resp.error());  
    }

    CUresult cuGetProcAddress (const char* symbol, void** pfn, int cudaVersion, cuuint64_t flags) {
        WapperQuery req;
        req.set_type(QueryType::cuGetProcAddress);

        cuGetProcAddressQuery query;
        query.set_symbol(string(symbol));
        query.set_version(cudaVersion);
        query.set_flags(static_cast<uint64_t>(flags));
        string query_serialized;
        query.SerializeToString(&query_serialized); 
        req.set_query(query_serialized);

        auto resp = sender_.send_and_recv<cuGetProcAddressResponse>(req);
        *pfn = reinterpret_cast<void*> (resp.pfn());
        return static_cast<CUresult>(resp.error());  
    }

    CUresult cuDriverGetVersion (int* driverVersion) {
        WapperQuery req;
        req.set_type(QueryType::cuDriverGetVersion);

        auto resp = sender_.send_and_recv<getVersionResponse>(req);
        *driverVersion = resp.version();
        return static_cast<CUresult>(resp.error());         
    }

    /**
    * CUDA runtime API
    */
    cudaError_t cudaDriverGetVersion(int* version) {
        WapperQuery req;
        req.set_type(QueryType::cudaDriverGetVersion);

        auto resp = sender_.send_and_recv<getVersionResponse>(req);
        *version = resp.version();
        return static_cast<cudaError_t>(resp.error());     
    }

    cudaError_t cudaRuntimeGetVersion(int* version) {
        WapperQuery req;
        req.set_type(QueryType::cudaRuntimeGetVersion);

        auto resp = sender_.send_and_recv<getVersionResponse>(req);
        *version = resp.version();
        return static_cast<cudaError_t>(resp.error());      
    }

    cudaError_t cudaGetLastError() {
        WapperQuery req;
        req.set_type(QueryType::cudaGetLastError);  
        
        // TODO: get message if error occurs
        sender_.send_async(req);
        return static_cast<cudaError_t>(0);  
    }


    cudaError_t cudaGetDeviceCount(int* count) {
        WapperQuery req;
        req.set_type(QueryType::cudaGetDeviceCount);
        
        auto resp = sender_.send_and_recv<cudaGetDeviceCountResponse>(req);
        *count = resp.count();
        return static_cast<cudaError_t>(resp.error());
    }

    cudaError_t cudaGetDevice(int* device){
        int err = 0;
        // use cache
        if (device_ >= 0) {
            *device = device_;
        }
        else{
            WapperQuery req;
            req.set_type(QueryType::cudaGetDevice);

            auto resp = sender_.send_and_recv<cudaGetDeviceResponse>(req);
            err = resp.error();
            device_ = resp.device();
            *device = resp.device();
        }

        return static_cast<cudaError_t>(err);
    }

    cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device) {
        WapperQuery req;
        req.set_type(QueryType::cudaGetDeviceProperties);

        cudaGetDevicePropertiesQuery query;
        query.set_device(device);
        string query_serialized;
        query.SerializeToString(&query_serialized); 
        req.set_query(query_serialized);

        auto resp = sender_.send_and_recv<cudaGetDevicePropertiesResponse>(req);
        memcpy(prop, &resp.prop()[0], resp.prop().length());
        return static_cast<cudaError_t>(resp.error());
    }

    cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device) {
        WapperQuery req;
        req.set_type(QueryType::cudaDeviceGetAttribute);

        cudaDeviceGetAttributeQuery query;
        query.set_device(device);
        query.set_attr(static_cast<int>(attr));
        string query_serialized;
        query.SerializeToString(&query_serialized); 
        req.set_query(query_serialized);

        auto resp = sender_.send_and_recv<cudaDeviceGetAttributeResponse>(req);
        *value = resp.value();
        return static_cast<cudaError_t>(resp.error());
    }

    cudaError_t cudaDeviceSynchronize() {
        WapperQuery req;
        req.set_type(QueryType::cudaDeviceSynchronize);
        auto resp = sender_.send_and_recv<genericResponse>(req);
        return static_cast<cudaError_t>(resp.error());
    }

    // memory
    cudaError_t	cudaMalloc(void** devPtr, size_t size) {
        WapperQuery req;
        req.set_type(QueryType::cudaMalloc);

        cudaMallocQuery query;
        query.set_size(size);
        string query_serialized;
        query.SerializeToString(&query_serialized); 
        req.set_query(query_serialized);

        auto resp = sender_.send_and_recv<cudaMallocResponse>(req);
        *devPtr = reinterpret_cast<void*>(resp.ptr());
        return static_cast<cudaError_t>(resp.error());
    }

    cudaError_t	cudaFree(void* devPtr) {
        WapperQuery req;
        req.set_type(QueryType::cudaFree);

        cudaFreeQuery query;
        query.set_ptr(reinterpret_cast<uint64_t>(devPtr));
        string query_serialized;
        query.SerializeToString(&query_serialized); 
        req.set_query(query_serialized);

        auto resp = sender_.send_and_recv<genericResponse>(req);
        return static_cast<cudaError_t>(resp.error());        
    }

    cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
        WapperQuery req;
        req.set_type(QueryType::cudaMemcpy);

        cudaMemcpyQuery query;
        query.set_src(reinterpret_cast<uint64_t>(src));
        query.set_dst(reinterpret_cast<uint64_t>(dst));
        query.set_count(count);
        query.set_kind(static_cast<int>(kind));

        if (kind == cudaMemcpyKind::cudaMemcpyHostToDevice) {
            query.set_payload(string(static_cast<const char*>(src), count));
        }
        
        string query_serialized;
        query.SerializeToString(&query_serialized); 
        req.set_query(query_serialized);

        auto resp = sender_.send_and_recv<cudaMemcpyResponse>(req);
        if (kind == cudaMemcpyKind::cudaMemcpyDeviceToHost) {
            memcpy(dst, &resp.payload()[0], resp.payload().length());
        }
        return static_cast<cudaError_t>(resp.error());        
    }

    cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0) {
        WapperQuery req;
        req.set_type(QueryType::cudaMemcpyAsync);

        cudaMemcpyAsyncQuery query;
        query.set_src(reinterpret_cast<uint64_t>(src));
        query.set_dst(reinterpret_cast<uint64_t>(dst));
        query.set_count(count);
        query.set_kind(static_cast<int>(kind));
        query.set_stream(string(reinterpret_cast<char*> (&stream), sizeof(stream)));

        if (kind == cudaMemcpyKind::cudaMemcpyHostToDevice) {
            query.set_payload(string(static_cast<const char*>(src), count));
        }
        
        string query_serialized;
        query.SerializeToString(&query_serialized); 
        req.set_query(query_serialized);

        sender_.send_async(req);
        return static_cast<cudaError_t>(0);
    }

    cudaError_t cudaMemsetAsync (void* devPtr, int value, size_t count, cudaStream_t stream = 0) {
        WapperQuery req;
        req.set_type(QueryType::cudaMemsetAsync);

        cudaMemsetAsyncQuery query;
        query.set_ptr(reinterpret_cast<uint64_t>(devPtr));
        query.set_value(value);
        query.set_count(count);
        query.set_stream(string(reinterpret_cast<char*> (&stream), sizeof(stream)));
        
        string query_serialized;
        query.SerializeToString(&query_serialized); 
        req.set_query(query_serialized);

        sender_.send_async(req);
        return static_cast<cudaError_t>(0); 
    }

    cudaError_t cudaGetSymbolAddress (void** devPtr, const void* symbol) {
        WapperQuery req;
        req.set_type(QueryType::cudaGetSymbolAddress);

        cudaGetSymbolAddressQuery query;
        query.set_symbol(reinterpret_cast<uint64_t> (symbol));
        string query_serialized;
        query.SerializeToString(&query_serialized); 
        req.set_query(query_serialized);

        auto resp = sender_.send_and_recv<cudaGetSymbolAddressResponse>(req);
        *devPtr = reinterpret_cast<void*>(resp.ptr());
        return static_cast<cudaError_t>(resp.error()); 
    }

    // stream management
    cudaError_t cudaStreamCreate(cudaStream_t* pStream) {
        WapperQuery req;
        req.set_type(QueryType::cudaStreamCreate);

        auto resp = sender_.send_and_recv<cudaStreamCreateResponse>(req);
        memcpy(pStream, &resp.stream()[0], resp.stream().length());
        return static_cast<cudaError_t>(resp.error());
    }

    cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags) {
        WapperQuery req;
        req.set_type(QueryType::cudaStreamCreateWithFlags);

        cudaStreamCreateWithFlagsQuery query;
        query.set_flags(flags);
        string query_serialized;
        query.SerializeToString(&query_serialized); 
        req.set_query(query_serialized);

        auto resp = sender_.send_and_recv<cudaStreamCreateResponse>(req);
        memcpy(pStream, &resp.stream()[0], resp.stream().length());
        return static_cast<cudaError_t>(resp.error());
    }

    cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pstream, unsigned int flags, int priority) {
        WapperQuery req;
        req.set_type(QueryType::cudaStreamCreateWithPriority);

        cudaStreamCreateWithPriorityQuery query;
        query.set_flags(flags);
        query.set_priority(priority);
        string query_serialized;
        query.SerializeToString(&query_serialized); 
        req.set_query(query_serialized);

        auto resp = sender_.send_and_recv<cudaStreamCreateResponse>(req);
        memcpy(pstream, &resp.stream()[0], resp.stream().length());
        return static_cast<cudaError_t>(resp.error());
    }

    cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
        WapperQuery req;
        req.set_type(QueryType::cudaStreamSynchronize);
        req.set_query(string(reinterpret_cast<char*> (&stream), sizeof(stream)));

        auto resp = sender_.send_and_recv<AsyncResponse>(req);
        for (auto &async_resp : resp.responses()) {
            memcpy(reinterpret_cast<void*>(async_resp.ptr()), &async_resp.payload()[0], async_resp.payload().length());
        }
        return static_cast<cudaError_t>(resp.error());
    }

    cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus* pCaptureStatus) {
        WapperQuery req;
        req.set_type(QueryType::cudaStreamIsCapturing);
        req.set_query(string(reinterpret_cast<char*> (&stream), sizeof(stream)));

        auto resp = sender_.send_and_recv<cudaStreamIsCapturingResponse>(req);
        *pCaptureStatus = static_cast<cudaStreamCaptureStatus>(resp.status());
        return static_cast<cudaError_t>(resp.error());
    }

    cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream, cudaStreamCaptureStatus* pCaptureStatus, unsigned long long* pId) {
        WapperQuery req;
        req.set_type(QueryType::cudaStreamGetCaptureInfo);
        req.set_query(string(reinterpret_cast<char*> (&stream), sizeof(stream)));

        auto resp = sender_.send_and_recv<cudaStreamGetCaptureInfoResponse>(req);
        *pCaptureStatus = static_cast<cudaStreamCaptureStatus>(resp.status());
        *pId = static_cast<unsigned long long>(resp.pid());
        return static_cast<cudaError_t>(resp.error());
    }

    cudaError_t cudaEventCreateWithFlags (cudaEvent_t* event, unsigned int flags) {
        WapperQuery req;
        req.set_type(QueryType::cudaEventCreateWithFlags);

        cudaEventCreateWithFlagsQuery query;
        query.set_flags(flags);
        string query_serialized;
        query.SerializeToString(&query_serialized); 
        req.set_query(query_serialized);

        auto resp = sender_.send_and_recv<cudaEventCreateWithFlagsResponse>(req);
        memcpy(event, &resp.event()[0], resp.event().length());
        return static_cast<cudaError_t>(resp.error());
    } 

    cudaError_t cudaEventQuery (cudaEvent_t event) {
        WapperQuery req;
        req.set_type(QueryType::cudaEventQuery);
        req.set_query(string(reinterpret_cast<char*> (&event), sizeof(event)));

        auto resp = sender_.send_and_recv<genericResponse>(req);
        log_->info("End cudaEventQuery");
        return static_cast<cudaError_t>(resp.error());   
    } 

    cudaError_t cudaEventRecord (cudaEvent_t event, cudaStream_t stream) {
        WapperQuery req;
        req.set_type(QueryType::cudaEventRecord);

        cudaEventRecordQuery query;
        query.set_event(string(reinterpret_cast<char*> (&event), sizeof(event)));
        query.set_stream(string(reinterpret_cast<char*> (&stream), sizeof(stream)));
        string query_serialized;
        query.SerializeToString(&query_serialized); 
        req.set_query(query_serialized);

        auto resp = sender_.send_and_recv<genericResponse>(req);
        log_->info("End cudaEventRecord");
        return static_cast<cudaError_t>(resp.error());   
    } 

    cudaError_t cudaLaunchKernel(string& func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream) {
        WapperQuery req;
        req.set_type(QueryType::cudaLaunchKernel);     

        cudaLaunchKernelQuery query;
        query.set_function(func);
        query.set_grid_dim_x(gridDim.x);
        query.set_grid_dim_y(gridDim.y);
        query.set_grid_dim_z(gridDim.z);
        query.set_block_dim_x(blockDim.x);
        query.set_block_dim_y(blockDim.y);
        query.set_block_dim_z(blockDim.z);
        query.set_shared_mem(sharedMem);
        query.set_stream(string(reinterpret_cast<char*> (&stream), sizeof(stream)));

        if (kernel_arg_info_.find(func) != kernel_arg_info_.end()) {
            for (int i = 0; i < kernel_arg_info_[func].size(); i++) {
                query.add_args(string(reinterpret_cast<char*> (args[i]), kernel_arg_info_[func][i]));
                // if (kernel_arg_info_[func][i] == 8) std::cout << "arg " << i << " : " << *((uint64_t*) args[i]) << std::endl;
            }

            string query_serialized;
            query.SerializeToString(&query_serialized); 
            req.set_query(query_serialized);

            sender_.send_async(req);
        }
        else {
            log_->warn("Cannot find kernel arg info for {}", func);
            std::cout << "Cannot find kernel arg info for " << func << std::endl;
        }
        return static_cast<cudaError_t>(0);
    }


private:
    CUDAClient() {
        auto id = sender_.get_client_id();
        string log_name = "log_client_" + std::to_string(id);
        string log_file = log_name + ".txt";
        log_ = spdlog::basic_logger_mt(log_name, log_file, true);
        log_->set_pattern("[%Y-%m-%d %T.%f] [%l] %v");
        log_->flush_on(spdlog::level::info);

        sender_.set_logger(log_);
        cuda_object_id_ = 1 << 20;

        // init kernel info
        std::ifstream file("/kernel_info.txt");
        string line; 
        while (std::getline(file, line)){
            std::stringstream ss(line);
            string item;
            vector<string> tokens;
            while (std::getline(ss, item, ',')) {
                tokens.push_back(item);
            }

            vector<size_t> kernel_arg;
            for (auto it = tokens.begin() + 2; it != tokens.end(); ++it) {
                kernel_arg.push_back(std::stoull(*it));
            }
            kernel_arg_info_[tokens[0]] = kernel_arg;
        }
        std::cout << "client init kernel_arg_info_ size: " << kernel_arg_info_.size() << std::endl;

        // init cuda-related cache
        device_ = -1;
    }

    inline uint64_t get_object_id() {
        return cuda_object_id_++;
    }

private:
    sender::AsyncSender sender_;

    logger log_;
    uint64_t cuda_object_id_;
    int device_;

    map<string, vector<size_t>> kernel_arg_info_;
};

} // namespace cuda_client
#endif  // INCLUDE_CUDA_CLIENT_HPP_
