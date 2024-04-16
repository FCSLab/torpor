#ifndef INCLUDE_CUDA_SERVER_HPP_
#define  INCLUDE_CUDA_SERVER_HPP_

#include <chrono>
#include <algorithm>
#include <stdexcept>

#include "cuda_common.hpp"
#include "utils.hpp"
#include "socket_helper.hpp"
#include "memory_manager.hpp"
#include "kernel_lookup.hpp"

namespace server {

struct AyncMessage {
    string function_id_; // for enabling multiple functions in future
    void* local_ptr_;
    size_t count_;
    void* remote_ptr_;
};

struct KernelMemoryAccessInfo {
    int arg_idx_;
    int offset_;
    bool is_model_access_; // access intermediate data otherwise
};

const int reqTimeout = 5000; // in milliseconds

const set<QueryType> wait_query_types = {QueryType::cuBLAS_cublasSgemm, QueryType::cuBLAS_cublasSgemmStridedBatched, 
                                            QueryType::cudnnConvolutionForward, QueryType::cudnnBatchNormalizationForwardInference, 
                                            QueryType::cudaMemcpyAsync};

const int LoadModelFlag_None = 0;
const int LoadModelFlag_LoadModel = 1;
const int LoadModelFlag_TrackAccess = 2;

const string non_active_func = "";

sf::safe_ptr<queue<int>> controller_sync_queue;
using ExecutorSyncQueue = sf::safe_ptr<queue<pair<int, string>>>;
using ExecutorRespBlockQueue = blocking_queue<int>;


class CUDAExecutor {
public:
    CUDAExecutor(int id, size_t budget, zmq::context_t* ctx): 
            id_(id),
            mem_manager_(budget),
            srv_pull_handler_(zmq::socket_t(*ctx, ZMQ_PULL)),
            pushers_(Pusher(ctx)){
        srv_pull_handler_.bind(get_server_addr(id));
        // pushers_.create(get_client_addr(0)); // default ipc

        string log_name = "log_executor_" + std::to_string(id_);
        string log_file = log_name + ".txt";
        log_ = spdlog::basic_logger_mt(log_name, log_file, true);
        log_->set_pattern("[%Y-%m-%d %T.%f] [%l] %v");

        auto log_level = get_env_var("EXECUTOR_LOG_LEVEL");
        auto cur_level = spdlog::level::info;

        if (log_level.size() > 0) {
            if (log_level == "debug") cur_level = spdlog::level::debug; // debug
            else if (log_level == "warn") cur_level = spdlog::level::warn; // warn
            else if (log_level == "err") cur_level = spdlog::level::err; // err
            else if (log_level == "off") cur_level = spdlog::level::off; // off
        }
        log_->set_level(cur_level);
        log_->flush_on(spdlog::level::info);
        
        mem_manager_.set_logger(log_);
        mem_manager_.set_server_id(id_);

        cudaCheck(cudaGetDeviceCount(&num_gpus_));

        executor_resp_block_queue_ = create_blocking_queue<int>();
  
        pollitems_ = {
            {static_cast<void*>(srv_pull_handler_), 0, ZMQ_POLLIN, 0} ,
        };
    }

    ~CUDAExecutor() {
        mem_manager_.destroy_trans_thread();
        cublasDestroy(cublas_handle_);
        cudnnDestroy(cudnn_handle_);
    }

public:
    void start(){
        cudaCheck(cudaSetDevice(id_));
        std::cout << "Server " << id_  << " -- start set device" << std::endl;

        // for (int i = 0; i < num_gpus_; i++) {
        //     if (i == id_) continue;
        //     int is_able;
        //     cudaDeviceCanAccessPeer(&is_able, id_, i);
        //     if (is_able) {
        //         auto err = cudaDeviceEnablePeerAccess(i, 0);
        //         std::cout << "Server " << id_  << " -- enable peer access " << i << " res " << err << std::endl;
        //     }
        // }

        mem_manager_.start_trans_thread(num_gpus_);

        cublasCreate(&cublas_handle_);
        cudnnCreate(&cudnn_handle_);

        active_func_ = non_active_func;
        load_model_flag_ = LoadModelFlag_None;

        cudaStreamCreateWithPriority(&default_stream_, cudaStreamNonBlocking, -1);
        cudaStreamCreateWithPriority(&secondary_stream_, cudaStreamNonBlocking, -1);

        req_start_ = std::chrono::system_clock::now();
        auto req_cur = std::chrono::system_clock::now();

        while (true) {
            kZmqUtil->poll(0, &pollitems_);

            // if (pollitems_[0].revents & ZMQ_POLLIN) {
            //     auto req_serialized = kZmqUtil->recv_string(&sig_handler_);
            //     SignalRequest req;
            //     req.ParseFromString(req_serialized);

            //     std::cout << "Server " << id_  << " -- Receive signal " << req.type() << std::endl;

            //     if (req.type() == RequestType::ExecuteAfterLoad) {
            //         auto func = req.function();
            //         auto origin_server = req.payload();
            //         if (origin_server.size() > 0 && stoi(origin_server) != id_) {
            //             std::cout << "Server " << id_  << " -- Receive func " << func << " for " << origin_server << std::endl;
            //         }
            //         else {
            //             active_func_ = func;
            //             load_model_flag_ = LoadModelFlag_LoadModel;
            //             mem_manager_.set_track_model_memory(active_func_);
            //         }
            //         pushers_.create(get_client_addr(stoi(func)));
            //         send_signal_ack(AckType::OK);
            //     }
            //     else if (req.type() == RequestType::Execute) {
            //         auto func = req.function();
            //         if (mem_manager_.has_model(func)) {
            //             active_func_ = func;
            //             send_signal_ack(AckType::OK);
            //             mem_manager_.sig_trans_thread(active_func_);
            //         }
            //         else {
            //             if (req.payload().size() > 0) {
            //                 int source_server_id = stoi(req.payload());
            //                 auto func_avail = mem_manager_.issue_load_model(func, source_server_id, true);
            //                 if (func_avail) {
            //                     active_func_ = func;
            //                     send_signal_ack(AckType::OK);
            //                     mem_manager_.sig_trans_thread(active_func_, true);
            //                 }
            //                 else {
            //                     std::cout << "Server " << id_  << " -- Inavail func " << func << " on source server " << source_server_id << std::endl;
            //                     send_signal_ack(AckType::Inavail);
            //                 }
            //             }
            //             else {
            //                 std::cout << "Server " << id_  << " -- Inavail func " << func << " and no source server found"<< std::endl;
            //                 send_signal_ack(AckType::Inavail);
            //             }
            //         }

            //     }
            //     else if (req.type() == RequestType::Load) {
            //         auto func = req.function();
            //         // TODO change to int
            //         int source_server_id = stoi(req.payload());
            //         mem_manager_.issue_load_model(func, source_server_id);
            //         send_signal_ack(AckType::OK);
            //     }
            //     else if (req.type() == RequestType::Unload) {
            //         // TODO remove model from local memory rather than evict it from cuda memory only
            //         auto func = req.function();
            //         mem_manager_.evict_model(func);
            //         // mem_manager_.sig_trans_thread(active_func_);
            //         // std::this_thread::sleep_for(std::chrono::milliseconds(100));

            //         send_signal_ack(AckType::OK);   
            //     }
            //     else if (req.type() == RequestType::ExecuteForRecord) {
            //         std::cout << "Decrepeted ExecuteForRecord signal" << std::endl;
            //         send_signal_ack(AckType::Inavail);
            //     }
            //     else if (req.type() == RequestType::Replay) {
            //         std::cout << "Decrepeted replay signal" << std::endl;
            //         send_signal_ack(AckType::Inavail);
            //     }
            // }

            // request from client
            if (pollitems_[0].revents & ZMQ_POLLIN) {
                auto req_serialized = kZmqUtil->recv_string(&srv_pull_handler_);

                QueryList query_list;
                query_list.ParseFromString(req_serialized);
                log_->debug("Receive API requests {}", query_list.queries_size());
                // log_->debug("API received {}", query_list.queries_size());
                for (auto &req : query_list.queries()){
                    send_query(req.type(), req.query(), req.client_id());
                }
                // log_->debug("API handled");
            }

            if (!executor_sync_queue_->empty()) {
                auto sig_func = executor_sync_queue_->front();
                executor_sync_queue_->pop();
                auto sig = sig_func.first;
                auto func = sig_func.second;

                log_->info("Receive signal {} for func {}", sig, func);

                if (sig == ExecutorSignal_Notify) {
                    pushers_.create(get_client_addr(stoi(func)));
                }
                else if (sig == ExecutorSignal_Startup) {
                    pushers_.create(get_client_addr(stoi(func)));
                    active_func_ = func;
                    load_model_flag_ = LoadModelFlag_LoadModel;
                    mem_manager_.set_track_model_memory(func);
                }
                else if (sig == ExecutorSignal_Execute) {
                    auto model_alloc_ready = mem_manager_.check_model_alloc_status(func);
                    if (!model_alloc_ready) {
                        std::cout << "Server " << id_  << " -- warning: model " << func << " not ready in allocation" << std::endl;
                        log_->warn("Model {} not ready for execution", func);
                    }
                    active_func_ = func;
                    load_model_flag_ = LoadModelFlag_None;

                    req_start_ = std::chrono::system_clock::now();
                }
                else if (sig == ExecutorSignal_Unload) {
                    mem_manager_.evict_model(func);
                }
                else if (sig >= ExecutorSignal_Load) {
                    int src_server_id = sig - ExecutorSignal_Load - 1;
                    // std::cout << "Server " << id_  << " -- load signal func " << func << " from server " << src_server_id << std::endl;
                    if (src_server_id >=0 && src_server_id != id_) {
                        mem_manager_.sig_trans_thread(func, src_server_id);
                    }
                    else {
                        // load from cpu
                        mem_manager_.sig_trans_thread(func);
                    }
                }
                executor_resp_block_queue_->enqueue(0);
            }

            if (active_func_ != non_active_func && load_model_flag_ == LoadModelFlag_None) {
                // check request timeout for Execute
                req_cur = std::chrono::system_clock::now();
                if (std::chrono::duration_cast<std::chrono::milliseconds>(req_cur - req_start_).count() > reqTimeout) {
                    log_->warn("Request timeout for func {}", active_func_);
                    send_complete_signal();
                }
            }

            // report_end = std::chrono::system_clock::now();
            // if (std::chrono::duration_cast<std::chrono::milliseconds>(report_end - report_start).count() > reportToSchedulerThreshold) {
            //     report_start = report_end;
            //     UpdateServerStatus msg;
            //     msg.set_server_id(id_);
            //     for (auto &model : mem_manager_.get_model_list()) {
            //         msg.add_functions(model);
            //     }
            //     // TODO update load
            //     string serialized;
            //     msg.SerializeToString(&serialized);
            //     kZmqUtil->send_string(serialized, &pushers_[get_scheduler_addr()]);
            // }
        }
    }

    void sig_exec_from_controller(int signal, string& func) {
        executor_sync_queue_->push({signal, func});
        executor_resp_block_queue_->dequeue();
    }

    void send_complete_signal() {
        // reset track flag
        if (load_model_flag_ == LoadModelFlag_TrackAccess) {
            load_model_flag_ = LoadModelFlag_None;
            mem_manager_.unset_track_model_memory(active_func_);
        }
        else {
            auto cur_time = std::chrono::system_clock::now();
            log_->info("Send complete signal for func {} time {}", active_func_, std::chrono::duration_cast<std::chrono::milliseconds>(cur_time - req_start_).count());
        }
        active_func_ = non_active_func;
        controller_sync_queue->push(id_);
    }

    void send_query(const QueryType &type, const string &query_string, int client_id) {
        switch(type) {
            // driver
            case QueryType::cuInit:
                cuInitService(query_string, client_id);
                break;
            case QueryType::cuDevicePrimaryCtxGetState:
                cuDevicePrimaryCtxGetStateService(query_string, client_id);
                break;         
            // case QueryType::cuGetProcAddress:
            //     cuGetProcAddressService(query_string, client_id);
            //     break;
            case QueryType::cuDriverGetVersion:
                cuDriverGetVersionService(query_string, client_id);
                break;
            // cublas
            case QueryType::cuBLAS_cublasCreate:
                cublasCreateService(query_string, client_id);
                break;    
            case QueryType::cuBLAS_cublasSetStream:
                cublasSetStreamService(query_string, client_id);
                break;   
            case QueryType::cuBLAS_cublasSetMathMode:
                cublasSetMathModeService(query_string, client_id);
                break;  
            case QueryType::cuBLAS_cublasGetMathMode:
                cublasGetMathModeService(query_string, client_id);
                break;   
            case QueryType::cuBLAS_cublasSgemm:
                cublasSgemmService(query_string, client_id);
                break;
            case QueryType::cuBLAS_cublasSgemmStridedBatched:
                cublasSgemmStridedBatchedService(query_string, client_id);
                break;
            // cudnn
            case QueryType::cudnnCreate:
                cudnnCreateService(query_string, client_id);
                break;   
            case QueryType::cudnnSetStream:
                cudnnSetStreamService(query_string, client_id);
                break;   
            case QueryType::cudnnCreateTensorDescriptor:
                cudnnCreateTensorDescriptorService(query_string, client_id);
                break;   
            case QueryType::cudnnSetTensorNdDescriptor:
                cudnnSetTensorNdDescriptorService(query_string, client_id);
                break;
            case QueryType::cudnnCreateFilterDescriptor:
                cudnnCreateFilterDescriptorService(query_string, client_id);
                break;   
            case QueryType::cudnnSetFilterNdDescriptor:
                cudnnSetFilterNdDescriptorService(query_string, client_id);
                break;  
            case QueryType::cudnnCreateConvolutionDescriptor:
                cudnnCreateConvolutionDescriptorService(query_string, client_id);
                break;
            case QueryType::cudnnSetConvolutionNdDescriptor:
                cudnnSetConvolutionNdDescriptorService(query_string, client_id);
                break;   
            case QueryType::cudnnSetConvolutionGroupCount:
                cudnnSetConvolutionGroupCountService(query_string, client_id);
                break;  
            case QueryType::cudnnSetConvolutionMathType:
                cudnnSetConvolutionMathTypeService(query_string, client_id);
                break;
            case QueryType::cudnnGetConvolutionForwardAlgorithm_v7:
                cudnnGetConvolutionForwardAlgorithm_v7Service(query_string, client_id);
                break;   
            case QueryType::cudnnConvolutionForward:
                cudnnConvolutionForwardService(query_string, client_id);
                break;     
            case QueryType::cudnnBatchNormalizationForwardInference:
                cudnnBatchNormalizationForwardInferenceService(query_string, client_id);
                break;     
            case QueryType::cudnnDestroyConvolutionDescriptor:
                cudnnDestroyConvolutionDescriptorService(query_string, client_id);
                break;
            case QueryType::cudnnDestroyFilterDescriptor:
                cudnnDestroyFilterDescriptorService(query_string, client_id);
                break;   
            case QueryType::cudnnDestroyTensorDescriptor:
                cudnnDestroyTensorDescriptorService(query_string, client_id);
                break;  
            // runtime version and device
            case QueryType::cudaGetLastError:
                cudaGetLastErrorService(query_string, client_id);
                break;
            case QueryType::cudaGetDeviceCount:
                cudaGetDeviceCountService(query_string, client_id);
                break;
            case QueryType::cudaGetDevice:
                cudaGetDeviceService(query_string, client_id);
                break;
            case QueryType::cudaGetDeviceProperties:
                cudaGetDevicePropertiesService(query_string, client_id);
                break;
            case QueryType::cudaDeviceGetAttribute:
                cudaDeviceGetAttributeService(query_string, client_id);
                break;
            case QueryType::cudaDeviceSynchronize:
                cudaDeviceSynchronizeService(query_string, client_id);
                break;
            // runtime memory
            case QueryType::cudaMalloc:
                cudaMallocService(query_string, client_id);
                break;
            case QueryType::cudaFree:
                cudaFreeService(query_string, client_id);
                break;
            case QueryType::cudaMemcpy:
                cudaMemcpyService(query_string, client_id);
                break;
            case QueryType::cudaMemcpyAsync:
                cudaMemcpyAsyncService(query_string, client_id);
                break;
            case QueryType::cudaMemsetAsync:
                cudaMemsetAsyncService(query_string, client_id);
                break;
            case QueryType::cudaGetSymbolAddress:
                cudaGetSymbolAddressService(query_string, client_id);
                break;
            // runtime stream
            case QueryType::cudaStreamCreate:
                cudaStreamCreateService(query_string, client_id);
                break;
            case QueryType::cudaStreamCreateWithFlags:
                cudaStreamCreateWithFlagsService(query_string, client_id);
                break;
            case QueryType::cudaStreamCreateWithPriority:
                cudaStreamCreateWithPriorityService(query_string, client_id);
                break;
            case QueryType::cudaStreamSynchronize:
                cudaStreamSynchronizeService(query_string, client_id);
                break;
            case QueryType::cudaStreamIsCapturing:
                cudaStreamIsCapturingService(query_string, client_id);
                break;
            case QueryType::cudaStreamGetCaptureInfo:
                cudaStreamGetCaptureInfoService(query_string, client_id);
                break;
            // runtime event
            case QueryType::cudaEventCreateWithFlags:
                cudaEventCreateWithFlagsService(query_string, client_id);
                break;
            case QueryType::cudaEventQuery:
                cudaEventQueryService(query_string, client_id);
                break;
            case QueryType::cudaEventRecord:
                cudaEventRecordService(query_string, client_id);
                break;
            case QueryType::cudaLaunchKernel:
                cudaLaunchKernelService(query_string, client_id);
                break;
        }
    }

    /** 
    * cublas
    */
    void cublasCreateService(const string& query_string, int client_id){
        // cublasHandle_t handle;
        // auto stat = cublasCreate(&handle);
        handleResponse resp;
        resp.set_error(static_cast<int> (0));
        resp.set_handle(string(reinterpret_cast<char*>(&cublas_handle_), sizeof(cublas_handle_)));

        send_response<handleResponse>(resp, client_id);
        log_->debug("Response: [cublas] cublasCreateService");
    }

    void cublasSetStreamService(const string& query_string, int client_id){
        setStreamQuery query;
        query.ParseFromString(query_string);

        cudaStream_t stream = get_stream(query.stream());
        auto stat = cublasSetStream(get_cublas_handle(query.handle()), stream);
        log_->debug("Async: [cublas] cublasSetStreamService");
    }

    void cublasSetMathModeService(const string& query_string, int client_id){
        cublasSetMathModeQuery query;
        query.ParseFromString(query_string);        
        
        // cublasHandle_t handle;
        // memcpy(&handle, &query.handle()[0], query.handle().length());
        auto mode = static_cast<cublasMath_t>(query.mode());
        auto stat = cublasSetMathMode(get_cublas_handle(query.handle()), mode);
        log_->debug("Async: [cublas] cublasSetMathModeService");
    }

    void cublasGetMathModeService(const string& query_string, int client_id){
        cublasMath_t mode;
        auto stat = cublasGetMathMode(get_cublas_handle(query_string), &mode);
        cublasGetMathModeResponse resp;
        resp.set_error(static_cast<int> (stat));
        resp.set_mode(static_cast<int> (mode));

        send_response<cublasGetMathModeResponse>(resp, client_id);
        log_->debug("Response: [cublas] cublasGetMathModeService");
    }

    void cublasSgemmService(const string& query_string, int client_id){
        cublasSgemmQuery query;
        query.ParseFromString(query_string);        
        
        auto transa = static_cast<cublasOperation_t>(query.transa());
        auto transb = static_cast<cublasOperation_t>(query.transb());
        auto alpha = query.alpha();
        auto beta = query.beta();

        auto A = reinterpret_cast<float*> (query.matrix_a());
        auto B = reinterpret_cast<float*> (query.matrix_b());
        auto C = reinterpret_cast<float*> (query.matrix_c());

        memoryCheck(mem_manager_.try_insert_model_access(A));
        A = (float*) mem_manager_.get_cuda_addr(active_func_, (void*) A);
        B = (float*) mem_manager_.get_cuda_addr(active_func_, (void*) B);
        C = (float*) mem_manager_.get_cuda_addr(active_func_, (void*) C);

        auto stat = cublasSgemm(get_cublas_handle(query.handle()), transa, transb, query.m(), query.n(), query.k(), &alpha, A, query.lda(), B, query.ldb(), &beta, C, query.ldc());
        log_->debug("Async: [cublas] cublasSgemmService");
    }

    void cublasSgemmStridedBatchedService(const string& query_string, int client_id){
        cublasSgemmStridedBatchedQuery query;
        query.ParseFromString(query_string);        

        auto transa = static_cast<cublasOperation_t>(query.transa());
        auto transb = static_cast<cublasOperation_t>(query.transb());
        auto alpha = query.alpha();
        auto beta = query.beta();

        auto A = reinterpret_cast<float*> (query.matrix_a());
        auto B = reinterpret_cast<float*> (query.matrix_b());
        auto C = reinterpret_cast<float*> (query.matrix_c());

        auto strideA = query.stride_a();
        auto strideB = query.stride_b();
        auto strideC = query.stride_c();
        auto batchCount = query.count();

        // memoryCheck(mem_manager_.try_insert_model_access(A));
        A = (float*) mem_manager_.get_cuda_addr(active_func_, (void*) A);
        B = (float*) mem_manager_.get_cuda_addr(active_func_, (void*) B);
        C = (float*) mem_manager_.get_cuda_addr(active_func_, (void*) C);

        auto stat = cublasSgemmStridedBatched(get_cublas_handle(query.handle()), transa, transb, query.m(), query.n(), query.k(), &alpha, A, query.lda(), strideA, B, query.ldb(), strideB, &beta, C, query.ldc(), strideC, batchCount);
        log_->debug("Async: [cublas] cublasSgemmStridedBatchedService");
        
    }


    /** 
    * cudnn
    */
    void cudnnCreateService(const string& query_string, int client_id){
        // cudnnHandle_t handle;
        // auto stat = cudnnCreate(&handle);
        handleResponse resp;
        resp.set_error(static_cast<int> (0));
        resp.set_handle(string(reinterpret_cast<char*>(&cudnn_handle_), sizeof(cudnn_handle_)));

        send_response<handleResponse>(resp, client_id);
        log_->debug("Response: [cudnn] cudnnCreateService");
    }

    void cudnnSetStreamService(const string& query_string, int client_id){
        setStreamQuery query;
        query.ParseFromString(query_string);

        cudaStream_t stream = get_stream(query.stream());        
        // cudnnHandle_t handle;
        // memcpy(&handle, &query.handle()[0], query.handle().length());

        cudnnCheck(cudnnSetStream(get_cudnn_handle(query.handle()), stream));
        log_->debug("Async: [cudnn] cudnnSetStreamService");
    }

    void cudnnCreateTensorDescriptorService (const string& query_string, int client_id){
        cudnnTensorDescriptor_t tensorDesc;
        cudnnCheck(cudnnCreateTensorDescriptor(&tensorDesc));
        insert_desc<cudnnTensorDescriptor_t>(query_string, tensorDesc, tensor_desc_map_);
        log_->debug("Async: [cudnn] cudnnCreateTensorDescriptorService");
    }

    void cudnnSetTensorNdDescriptorService(const string& query_string, int client_id){
        cudnnSetTensorNdDescriptorQuery query;
        query.ParseFromString(query_string);

        // memcpy(&tensorDesc, &query.cudnn_descriptor()[0], query.cudnn_descriptor().length());

        cudnnTensorDescriptor_t tensorDesc = get_desc<cudnnTensorDescriptor_t>(query.cudnn_descriptor(), tensor_desc_map_);

        auto type = static_cast<cudnnDataType_t>(query.type());
        auto dims = query.dims();
        int dimA[dims];
        int strideA[dims];
        for (int i = 0; i < dims; i++) {
            dimA[i] = query.dim_a(i);
            strideA[i] = query.stride_a(i);
        }

        cudnnCheck(cudnnSetTensorNdDescriptor(tensorDesc, type, dims, dimA, strideA));
        log_->debug("Async: [cudnn] cudnnSetTensorNdDescriptorService");
    }

    void cudnnCreateFilterDescriptorService(const string& query_string, int client_id){
        cudnnFilterDescriptor_t filterDesc;
        cudnnCheck(cudnnCreateFilterDescriptor(&filterDesc));
        insert_desc<cudnnFilterDescriptor_t>(query_string, filterDesc, filter_desc_map_);

        log_->debug("Async: [cudnn] cudnnCreateFilterDescriptorService");
    }

    void cudnnSetFilterNdDescriptorService(const string& query_string, int client_id){
        cudnnSetFilterNdDescriptorQuery query;
        query.ParseFromString(query_string);

        cudnnFilterDescriptor_t desc = get_desc<cudnnFilterDescriptor_t>(query.cudnn_descriptor(), filter_desc_map_);

        auto type = static_cast<cudnnDataType_t>(query.type());
        auto format = static_cast<cudnnTensorFormat_t>(query.format());
        
        auto dims = query.dims();
        int filterDimA[dims];
        for (int i = 0; i < dims; i++) {
            filterDimA[i] = query.dim_a(i);
        }

        cudnnCheck(cudnnSetFilterNdDescriptor(desc, type, format, dims, filterDimA));
        log_->debug("Async: [cudnn] cudnnSetFilterNdDescriptorService");
    }

    void cudnnCreateConvolutionDescriptorService(const string& query_string, int client_id){
        cudnnConvolutionDescriptor_t desc;

        cudnnCheck(cudnnCreateConvolutionDescriptor(&desc));
        insert_desc<cudnnConvolutionDescriptor_t>(query_string, desc, conv_desc_map_);

        log_->debug("Async: [cudnn] cudnnCreateConvolutionDescriptorService");
    }

    void cudnnSetConvolutionNdDescriptorService(const string& query_string, int client_id){
        cudnnSetConvolutionNdDescriptorQuery query;
        query.ParseFromString(query_string);

        cudnnConvolutionDescriptor_t desc = get_desc<cudnnConvolutionDescriptor_t>(query.cudnn_descriptor(), conv_desc_map_);

        auto type = static_cast<cudnnDataType_t>(query.type());
        auto mode = static_cast<cudnnConvolutionMode_t>(query.mode());
        
        auto length = query.length();
        int padA[length];
        int filterStrideA[length];
        int dilationA[length];
        for (int i = 0; i < length; i++) {
            padA[i] = query.pad_a(i);
            filterStrideA[i] = query.stride_a(i);
            dilationA[i] = query.dilation_a(i);
        }

        cudnnCheck(cudnnSetConvolutionNdDescriptor(desc, length, padA, filterStrideA, dilationA, mode, type));
        log_->debug("Async: [cudnn] cudnnSetConvolutionNdDescriptorService");
    }

    void cudnnSetConvolutionGroupCountService(const string& query_string, int client_id){
        cudnnSetConvolutionGroupCountQuery query;
        query.ParseFromString(query_string);
        
        cudnnConvolutionDescriptor_t desc = get_desc<cudnnConvolutionDescriptor_t>(query.cudnn_descriptor(), conv_desc_map_);
        cudnnCheck(cudnnSetConvolutionGroupCount(desc, query.count()));
        log_->debug("Async: [cudnn] cudnnSetConvolutionGroupCountService");
    }

    void cudnnSetConvolutionMathTypeService(const string& query_string, int client_id){
        cudnnSetConvolutionMathTypeQuery query;
        query.ParseFromString(query_string);
        
        cudnnConvolutionDescriptor_t desc = get_desc<cudnnConvolutionDescriptor_t>(query.cudnn_descriptor(), conv_desc_map_);
        auto type = static_cast<cudnnMathType_t> (query.type());
        cudnnCheck(cudnnSetConvolutionMathType(desc, type));
        log_->debug("Async: [cudnn] cudnnSetConvolutionMathTypeService");
    }

    void cudnnGetConvolutionForwardAlgorithm_v7Service(const string& query_string, int client_id){
        cudnnGetConvolutionForwardAlgorithm_v7Query query;
        query.ParseFromString(query_string);

        cudnnTensorDescriptor_t xDesc = get_desc<cudnnTensorDescriptor_t>(query.x_desc(), tensor_desc_map_);
        cudnnFilterDescriptor_t wDesc = get_desc<cudnnFilterDescriptor_t>(query.w_desc(), filter_desc_map_);
        cudnnConvolutionDescriptor_t convDesc = get_desc<cudnnConvolutionDescriptor_t>(query.conv_desc(), conv_desc_map_);
        cudnnTensorDescriptor_t yDesc = get_desc<cudnnTensorDescriptor_t>(query.y_desc(), tensor_desc_map_);

        int requestedAlgoCount = query.count();
        int returnedAlgoCount;
        cudnnConvolutionFwdAlgoPerf_t perfResults[requestedAlgoCount];
        auto stat = cudnnGetConvolutionForwardAlgorithm_v7(get_cudnn_handle(query.handle()), xDesc, wDesc, convDesc, yDesc, requestedAlgoCount, &returnedAlgoCount, perfResults);
        
        cudnnGetConvolutionForwardAlgorithm_v7Response resp;
        resp.set_error(static_cast<int> (stat));
        resp.set_count(returnedAlgoCount);
        resp.set_results(string(reinterpret_cast<char*>(perfResults), sizeof(perfResults)));

        send_response<cudnnGetConvolutionForwardAlgorithm_v7Response>(resp, client_id);
        log_->debug("Response: [cudnn] cudnnGetConvolutionForwardAlgorithm_v7Service requested {}, returned {}", requestedAlgoCount, returnedAlgoCount);
    }

    void cudnnConvolutionForwardService(const string& query_string, int client_id){
        cudnnConvolutionForwardQuery query;
        query.ParseFromString(query_string);

        cudnnTensorDescriptor_t xDesc = get_desc<cudnnTensorDescriptor_t>(query.x_desc(), tensor_desc_map_);
        cudnnFilterDescriptor_t wDesc = get_desc<cudnnFilterDescriptor_t>(query.w_desc(), filter_desc_map_);
        cudnnConvolutionDescriptor_t convDesc = get_desc<cudnnConvolutionDescriptor_t>(query.conv_desc(), conv_desc_map_);
        cudnnTensorDescriptor_t yDesc = get_desc<cudnnTensorDescriptor_t>(query.y_desc(), tensor_desc_map_);

        auto alpha = query.alpha();
        auto beta = query.beta();
        auto algo = static_cast<cudnnConvolutionFwdAlgo_t> (query.algo());
        auto x = reinterpret_cast<void*> (query.x());
        auto w = reinterpret_cast<void*> (query.w());
        auto y = reinterpret_cast<void*> (query.y());
        auto workspace = reinterpret_cast<void*> (query.workspace());
        size_t workspace_size = query.workspace_size();

        memoryCheck(mem_manager_.try_insert_model_access(w));
        w = mem_manager_.get_cuda_addr(active_func_, w);
        x = mem_manager_.get_cuda_addr(active_func_, x);
        y = mem_manager_.get_cuda_addr(active_func_, y);
        // workspace can be 0 
        workspace = mem_manager_.get_cuda_addr(active_func_, workspace);

        cudnnCheck(cudnnConvolutionForward(get_cudnn_handle(query.handle()), &alpha, xDesc, x, wDesc, w, convDesc, algo, workspace, workspace_size, &beta, yDesc, y));
        log_->debug("Async: [cudnn] cudnnConvolutionForwardService");
    }

    void cudnnBatchNormalizationForwardInferenceService(const string& query_string, int client_id){
        cudnnBatchNormalizationForwardInferenceQuery query;
        query.ParseFromString(query_string);

        cudnnTensorDescriptor_t xDesc = get_desc<cudnnTensorDescriptor_t>(query.x_desc(), tensor_desc_map_);
        cudnnTensorDescriptor_t yDesc = get_desc<cudnnTensorDescriptor_t>(query.y_desc(), tensor_desc_map_);
        cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc = get_desc<cudnnTensorDescriptor_t>(query.bn_desc(), tensor_desc_map_);

        auto epsilon = query.epsilon();
        auto mode = static_cast<cudnnBatchNormMode_t> (query.mode());
        auto alpha = query.alpha();
        auto beta = query.beta();

        auto x = reinterpret_cast<void*> (query.x());
        auto y = reinterpret_cast<void*> (query.y());
        auto bnScale = reinterpret_cast<void*> (query.bn_scale());
        auto bnBias = reinterpret_cast<void*> (query.bn_bias());
        auto estimatedMean = reinterpret_cast<void*> (query.es_mean());
        auto estimatedVariance = reinterpret_cast<void*> (query.es_var());

        memoryCheck(mem_manager_.try_insert_model_access(bnScale));
        memoryCheck(mem_manager_.try_insert_model_access(bnBias));
        memoryCheck(mem_manager_.try_insert_model_access(estimatedMean));
        memoryCheck(mem_manager_.try_insert_model_access(estimatedVariance));

        bnScale = mem_manager_.get_cuda_addr(active_func_, bnScale);
        bnBias = mem_manager_.get_cuda_addr(active_func_, bnBias);
        estimatedMean = mem_manager_.get_cuda_addr(active_func_, estimatedMean);
        estimatedVariance = mem_manager_.get_cuda_addr(active_func_, estimatedVariance);
        x = mem_manager_.get_cuda_addr(active_func_, x);
        y = mem_manager_.get_cuda_addr(active_func_, y);

        cudnnCheck(cudnnBatchNormalizationForwardInference(get_cudnn_handle(query.handle()), mode, &alpha, &beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, estimatedMean, estimatedVariance, epsilon));
        log_->debug("Async: [cudnn] cudnnBatchNormalizationForwardInferenceService");
    }

    void cudnnDestroyConvolutionDescriptorService(const string& query_string, int client_id){
        cudnnConvolutionDescriptor_t desc = get_desc<cudnnConvolutionDescriptor_t>(query_string, conv_desc_map_);
        // memcpy(&desc, &query_string[0], query_string.length());
        cudnnCheck(cudnnDestroyConvolutionDescriptor(desc));
        conv_desc_map_.erase(query_string);

        log_->debug("Async: [cudnn] cudnnDestroyConvolutionDescriptorService");
    }

    void cudnnDestroyFilterDescriptorService(const string& query_string, int client_id){
        cudnnFilterDescriptor_t desc = get_desc<cudnnFilterDescriptor_t>(query_string, filter_desc_map_);
        // memcpy(&desc, &query_string[0], query_string.length());
        cudnnCheck(cudnnDestroyFilterDescriptor(desc));
        filter_desc_map_.erase(query_string);
        log_->debug("Async: [cudnn] cudnnDestroyFilterDescriptorService");
    }

    void cudnnDestroyTensorDescriptorService(const string& query_string, int client_id){
        cudnnTensorDescriptor_t desc = get_desc<cudnnTensorDescriptor_t>(query_string, tensor_desc_map_);
        // memcpy(&desc, &query_string[0], query_string.length());

        cudnnCheck(cudnnDestroyTensorDescriptor(desc));
        tensor_desc_map_.erase(query_string);
        log_->debug("Async: [cudnn] cudnnDestroyTensorDescriptorService");
    }

    /**
    * cuda driver
    */
    void cuInitService(const string& query_string, int client_id){
        cuInitQuery query;
        query.ParseFromString(query_string);

        auto res = cuInit(query.flags());

        genericResponse resp;
        resp.set_error(static_cast<int> (res));

        send_response<genericResponse>(resp, client_id);
        log_->debug("Response: [driver] cuInitService");
    }

    void cuDevicePrimaryCtxGetStateService(const string& query_string, int client_id){
        cuDevicePrimaryCtxGetStateQuery query;
        query.ParseFromString(query_string);

        auto dev = static_cast<CUdevice> (query.device());
        unsigned int ctx_flags;
        int ctx_is_active = 0;

        auto res = cuDevicePrimaryCtxGetState(dev, &ctx_flags, &ctx_is_active);

        cuDevicePrimaryCtxGetStateResponse resp;
        resp.set_flags(ctx_flags);
        resp.set_active(ctx_is_active);
        resp.set_error(static_cast<int> (res));

        send_response<cuDevicePrimaryCtxGetStateResponse>(resp, client_id);
        log_->debug("Response: [driver] cuDevicePrimaryCtxGetStateService");
    }

    // void cuGetProcAddressService(const string& query_string, int client_id){
    //     cuGetProcAddressQuery query;
    //     query.ParseFromString(query_string);

    //     auto symbol = query.symbol().c_str();
    //     auto flags = static_cast<cuuint64_t>(query.flags());
    //     void* pfn;
    //     auto res = cuGetProcAddress(symbol, &pfn, query.version(), flags);

    //     cuGetProcAddressResponse resp;
    //     resp.set_pfn(reinterpret_cast<uint64_t>(pfn));
    //     resp.set_error(static_cast<int> (res));

    //     send_response<cuGetProcAddressResponse>(resp);
    //     log_->debug("Response: [driver] cuGetProcAddressService");
    // }

    void cuDriverGetVersionService(const string& query_string, int client_id){
        int version;
        auto err = cuDriverGetVersion(&version);
        getVersionResponse resp;
        resp.set_version(version);
        resp.set_error(static_cast<int> (err));

        send_response<getVersionResponse>(resp, client_id);
        log_->debug("Response: [driver] cuDriverGetVersionService");
    }

    /** 
    * cuda runtime
    */
    void cudaDriverGetVersionService(const string& query_string, int client_id){
        int version;
        auto err = cudaDriverGetVersion(&version);
        getVersionResponse resp;
        resp.set_version(version);
        resp.set_error(static_cast<int> (err));

        send_response<getVersionResponse>(resp, client_id);
        log_->debug("Response: cudaDriverGetVersionService");
    }

    void cudaRuntimeGetVersionService(const string& query_string, int client_id){
        int version;
        auto err = cudaRuntimeGetVersion(&version);
        getVersionResponse resp;
        resp.set_version(version);
        resp.set_error(static_cast<int> (err));

        send_response<getVersionResponse>(resp, client_id);
        log_->debug("Response: cudaDriverGetVersionService");
    }

    void cudaGetLastErrorService(const string& query_string, int client_id){
        cudaCheck(cudaGetLastError());
        // genericResponse resp;
        // resp.set_error(static_cast<int> (err));

        // send_response<genericResponse>(resp);
        log_->debug("Async: cudaGetLastErrorService");
    }

    void cudaGetDeviceCountService(const string& query_string, int client_id){
        int count = 0;
        auto err = cudaGetDeviceCount(&count);
        cudaGetDeviceCountResponse resp;
        resp.set_count(count);
        resp.set_error(static_cast<int> (err));

        send_response<cudaGetDeviceCountResponse>(resp, client_id);
        log_->debug("Response: cudaGetDeviceCountService");
    }

    void cudaGetDeviceService(const string& query_string, int client_id){
        int device = 0;
        auto err = cudaGetDevice(&device);
        cudaGetDeviceResponse resp;
        resp.set_device(device);
        resp.set_error(static_cast<int> (err));

        send_response<cudaGetDeviceResponse>(resp, client_id);
        log_->debug("Response: cudaGetDeviceService");
    }

    void cudaGetDevicePropertiesService(const string& query_string, int client_id){
        cudaGetDevicePropertiesQuery query;
        query.ParseFromString(query_string);

        cudaDeviceProp device_prop;
        auto err = cudaGetDeviceProperties(&device_prop, query.device());

        cudaGetDevicePropertiesResponse resp;
        resp.set_error(static_cast<int> (err));
        resp.set_prop(string(reinterpret_cast<char*>(&device_prop), sizeof(device_prop)));

        send_response<cudaGetDevicePropertiesResponse>(resp, client_id);
        log_->debug("Response: cudaGetDevicePropertiesService");
    }


    void cudaDeviceGetAttributeService(const string& query_string, int client_id){
        cudaDeviceGetAttributeQuery query;
        query.ParseFromString(query_string);

        auto attr = static_cast<cudaDeviceAttr>(query.attr());
        auto device = query.device();
        int value;
        auto err = cudaDeviceGetAttribute(&value, attr, device);

        cudaDeviceGetAttributeResponse resp;
        resp.set_error(static_cast<int> (err));
        resp.set_value(value);

        send_response<cudaDeviceGetAttributeResponse>(resp, client_id);
        log_->debug("Response: cudaDeviceGetAttributeService");
    }
    
    void cudaDeviceSynchronizeService(const string& query_string, int client_id){
        auto err = cudaDeviceSynchronize();
        genericResponse resp;
        resp.set_error(static_cast<int> (err));

        send_response<genericResponse>(resp, client_id);
        log_->debug("Response: cudaDeviceSynchronizeService");

        // walkaround: signal to specify following memory requests not belonging to models
        load_model_flag_ = LoadModelFlag_TrackAccess;
        mem_manager_.init_model_readiness(active_func_);
    }   

    // memory management
    void cudaMallocService(const string& query_string, int client_id){
        cudaMallocQuery query;
        query.ParseFromString(query_string);

        auto size = query.size();
        auto ptr = mem_manager_.cudaMallocBackend(size, load_model_flag_==LoadModelFlag_LoadModel, active_func_);

        cudaMallocResponse resp;
        resp.set_ptr(reinterpret_cast<uint64_t>(ptr));
        resp.set_error(0);
        
        send_response<cudaMallocResponse>(resp, client_id);
        log_->debug("Response: cudaMallocService");
    }

    void cudaFreeService(const string& query_string, int client_id){
        cudaFreeQuery query;
        query.ParseFromString(query_string);

        auto ptr = reinterpret_cast<void*>(query.ptr());
        mem_manager_.cudaFreeBackend(ptr, active_func_);

        genericResponse resp;
        resp.set_error(0);

        send_response<genericResponse>(resp, client_id);
        log_->debug("Response: cudaFreeService");
    }

    void cudaMemcpyService(const string& query_string, int client_id){
        cudaMemcpyQuery query;
        query.ParseFromString(query_string);

        auto kind = static_cast<cudaMemcpyKind>(query.kind());

        cudaMemcpyResponse resp;
        if (kind == cudaMemcpyKind::cudaMemcpyHostToDevice) {
            auto dst = reinterpret_cast<void*>(query.dst());
            mem_manager_.cudaMemcpyBackendHtoD(dst, query.payload(), active_func_);
        }
        else if (kind == cudaMemcpyKind::cudaMemcpyDeviceToHost) {
            auto src = reinterpret_cast<void*>(query.src());
            void* local_dst = malloc(query.count());
            mem_manager_.cudaMemcpyBackendDtoH(local_dst, src, query.count(), active_func_);
            resp.set_payload(string(static_cast<char*>(local_dst), query.count()));
            free(local_dst);
        } 
        else if (kind == cudaMemcpyKind::cudaMemcpyDeviceToDevice) {
            auto src = reinterpret_cast<void*>(query.src());
            auto dst = reinterpret_cast<void*>(query.dst());
            mem_manager_.cudaMemcpyBackendDtoD(dst, src, query.count(), active_func_);
        } 
        else {
            log_->warn("Unsupported cudaMemcpyKind {}", kind);
            std::cout << "Unsupported cudaMemcpyKind " << kind << std::endl;
        }
        resp.set_error(0);

        send_response<cudaMemcpyResponse>(resp, client_id);
        log_->debug("Response: cudaMemcpyService");
    }

    void cudaMemcpyAsyncService(const string& query_string, int client_id){
        cudaMemcpyAsyncQuery query;
        query.ParseFromString(query_string);

        auto kind = static_cast<cudaMemcpyKind>(query.kind());
        cudaStream_t stream = get_stream(query.stream());

        if (kind == cudaMemcpyKind::cudaMemcpyHostToDevice) {
            auto dst = reinterpret_cast<void*>(query.dst());
            mem_manager_.cudaMemcpyAsyncBackendHtoD(dst, query.payload(), stream, load_model_flag_==LoadModelFlag_LoadModel, active_func_);
        }
        else if (kind == cudaMemcpyKind::cudaMemcpyDeviceToHost) {
            auto src = reinterpret_cast<void*>(query.src());
            void* local_dst = malloc(query.count());
            mem_manager_.cudaMemcpyAsyncBackendDtoH(local_dst, src, query.count(), stream, active_func_);
            stream_async_msg_.push_back({active_func_, local_dst, query.count(), reinterpret_cast<void*>(query.dst())});
            
            send_complete_signal();
        } 
        else if (kind == cudaMemcpyKind::cudaMemcpyDeviceToDevice) {
            auto src = reinterpret_cast<void*>(query.src());
            auto dst = reinterpret_cast<void*>(query.dst());
            mem_manager_.cudaMemcpyAsyncBackendDtoD(dst, src, query.count(), stream, active_func_);
        } 
        else {
            std::cout << "Unsupported kind " << kind << std::endl;
        }

        log_->debug("Async: cudaMemcpyAsyncService");
    }

    void cudaMemsetAsyncService(const string& query_string, int client_id){
        cudaMemsetAsyncQuery query;
        query.ParseFromString(query_string);
        
        auto ptr = reinterpret_cast<void*>(query.ptr());
        cudaStream_t stream = get_stream(query.stream());

        mem_manager_.cudaMemsetAsyncBackend(ptr, query.value(), query.count(), stream, active_func_);
        
        log_->debug("Async: cudaMemsetAsyncService");
    }

    void cudaGetSymbolAddressService(const string& query_string, int client_id){
        cudaGetSymbolAddressQuery query;
        query.ParseFromString(query_string);

        auto symbol = reinterpret_cast<void*>(query.symbol());
        void* ptr;
        auto err = cudaGetSymbolAddress(&ptr, symbol);

        cudaGetSymbolAddressResponse resp;
        resp.set_ptr(reinterpret_cast<uint64_t>(ptr));
        resp.set_error(static_cast<int> (err));

        send_response<cudaGetSymbolAddressResponse>(resp, client_id);
        log_->debug("Response: cudaGetSymbolAddressService");
    }

    // stream
    void cudaStreamCreateService(const string& query_string, int client_id){
        cudaStream_t stream = create_stream(cudaStreamDefault, 0);
        // auto err = cudaStreamCreate(&stream);

        cudaStreamCreateResponse resp;
        resp.set_stream(string(reinterpret_cast<char*>(&stream), sizeof(stream)));
        resp.set_error(static_cast<int> (0));
        
        send_response<cudaStreamCreateResponse>(resp, client_id);
        log_->debug("Response: cudaStreamCreateService");
    }

    void cudaStreamCreateWithFlagsService(const string& query_string, int client_id){
        cudaStreamCreateWithFlagsQuery query;
        query.ParseFromString(query_string);

        auto flags = query.flags();
        cudaStream_t stream = create_stream(flags, 0);
        // auto err = cudaStreamCreateWithFlags(&stream, flags);

        cudaStreamCreateResponse resp;
        resp.set_stream(string(reinterpret_cast<char*>(&stream), sizeof(stream)));
        resp.set_error(static_cast<int> (0));
        
        send_response<cudaStreamCreateResponse>(resp, client_id);
        log_->debug("Response: cudaStreamCreateWithFlagsService");
    }
    
    void cudaStreamCreateWithPriorityService(const string& query_string, int client_id){
        cudaStreamCreateWithPriorityQuery query;
        query.ParseFromString(query_string);

        auto flags = query.flags();
        auto priority = query.priority();
        
        cudaStream_t stream = create_stream(flags, priority);

        cudaStreamCreateResponse resp;
        resp.set_error(static_cast<int> (0));
        resp.set_stream(string(reinterpret_cast<char*>(&stream), sizeof(stream)));
        
        send_response<cudaStreamCreateResponse>(resp, client_id);
        log_->debug("Response: cudaStreamCreateWithPriority");
    }

    void cudaStreamSynchronizeService(const string& query_string, int client_id){
        cudaStream_t stream = get_stream(query_string);

        cudaCheck(cudaStreamSynchronize(stream));
        AsyncResponse resp;
        resp.set_error(static_cast<int> (0));

        for (auto &async_msg : stream_async_msg_) {
            auto async_resp = resp.add_responses();
            async_resp->set_ptr(reinterpret_cast<uint64_t>(async_msg.remote_ptr_));
            async_resp->set_payload(string(static_cast<char*>(async_msg.local_ptr_), async_msg.count_));
            free(async_msg.local_ptr_);
        }
        stream_async_msg_.clear();

        send_response<AsyncResponse>(resp, client_id);
        log_->debug("Response: cudaStreamSynchronizeService");
    }

    void cudaStreamIsCapturingService(const string& query_string, int client_id){
        cudaStream_t stream = get_stream(query_string);

        cudaStreamCaptureStatus status;     
        auto err = cudaStreamIsCapturing(stream, &status);

        cudaStreamIsCapturingResponse resp;
        resp.set_error(static_cast<int> (err));
        resp.set_status(static_cast<int> (status));

        send_response<cudaStreamIsCapturingResponse>(resp, client_id);
        log_->debug("Response: cudaStreamIsCapturingService");
    }

    void cudaStreamGetCaptureInfoService(const string& query_string, int client_id){
        cudaStream_t stream = get_stream(query_string);

        cudaStreamCaptureStatus status; 
        unsigned long long pid;  
        auto err = cudaStreamGetCaptureInfo(stream, &status, &pid);
        
        cudaStreamGetCaptureInfoResponse resp;
        resp.set_error(static_cast<int> (err));
        resp.set_status(static_cast<int> (status));
        resp.set_pid(static_cast<uint64_t> (pid));

        send_response<cudaStreamGetCaptureInfoResponse>(resp, client_id);
        log_->debug("Response: cudaStreamGetCaptureInfoService");
    }
    
    // event 
    void cudaEventCreateWithFlagsService(const string& query_string, int client_id){
        cudaEventCreateWithFlagsQuery query;
        query.ParseFromString(query_string);

        auto flags = query.flags();
        cudaEvent_t event;
        auto err = cudaEventCreateWithFlags(&event, flags);

        cudaEventCreateWithFlagsResponse resp;
        resp.set_event(string(reinterpret_cast<char*>(&event), sizeof(event)));
        resp.set_error(static_cast<int> (err));

        send_response<cudaEventCreateWithFlagsResponse>(resp, client_id);
        log_->debug("Response: cudaEventCreateWithFlagsService");
    }

    void cudaEventQueryService(const string& query_string, int client_id){
        cudaEvent_t event;
        memcpy(&event, &query_string[0], query_string.length());  

        auto err = cudaEventQuery(event);

        genericResponse resp;
        resp.set_error(static_cast<int> (err));
        send_response<genericResponse>(resp, client_id);
        log_->debug("Response: cudaEventQueryService");
    }

    void cudaEventRecordService (const string& query_string, int client_id){
        cudaEventRecordQuery query;
        query.ParseFromString(query_string);

        cudaEvent_t event;
        memcpy(&event, &query.event()[0], query.event().length());  
        cudaStream_t stream = get_stream(query.stream());

        cudaCheck(cudaEventRecord(event, stream));
        genericResponse resp;
        resp.set_error(static_cast<int> (0));

        send_response<genericResponse>(resp, client_id);
        log_->debug("Response: cudaEventRecordService");
    }

    void cudaLaunchKernelService(const string& query_string, int client_id){
        cudaLaunchKernelQuery query;
        query.ParseFromString(query_string);

        if (lookup().find(query.function()) != lookup().end()) {
            auto ptr = lookup()[query.function()];
            dim3 gridDim(query.grid_dim_x(), query.grid_dim_y(), query.grid_dim_z());
            dim3 blockDim(query.block_dim_x(), query.block_dim_y(), query.block_dim_z());

            // std::cout << "cudaKernel " << query.function();

            void* args[query.args_size()];
            for (int i = 0; i < query.args_size(); i++) {
                args[i] = malloc(query.args(i).length());
                memcpy(args[i], &query.args(i)[0], query.args(i).length());
            }

            // first execution
            if (load_model_flag_ == LoadModelFlag_TrackAccess) {
                if (kernel_model_access_loc().find(active_func_) == kernel_model_access_loc().end()) {
                    kernel_model_access_loc()[active_func_] = {};
                }
                if (kernel_model_access_loc()[active_func_].find(ptr) == kernel_model_access_loc()[active_func_].end()) {
                    kernel_model_access_loc()[active_func_][ptr] = {};
                }
                for (int i = 0; i < query.args_size(); i++) {
                    if (query.args(i).length() % sizeof(uint64_t) == 0) {
                        for (int j = 0; j <  query.args(i).length() / sizeof(uint64_t); j++) {
                            if (mem_manager_.is_model_address(active_func_, (void*) ((uint64_t*) args[i])[j])) {
                                memoryCheck(mem_manager_.try_insert_model_access((void*) ((uint64_t*) args[i])[j]));
                                kernel_model_access_loc()[active_func_][ptr].insert({i, j});
                            }
                            else if (mem_manager_.valid_memory_address(active_func_, (void*) ((uint64_t*) args[i])[j])) {
                                kernel_model_access_loc()[active_func_][ptr].insert({i, j});
                            }
                        }
                    }
                }
            }
            else {
                for (auto &info : kernel_model_access_loc()[active_func_][ptr]) {
                    ((uint64_t*) args[info.first])[info.second] = (uint64_t) mem_manager_.get_cuda_addr(active_func_, (void*) ((uint64_t*) args[info.first])[info.second]);
                }
            }

            // if (kernel_model_access_loc().find(ptr) == kernel_model_access_loc().end()) {
            //     // set model access loc for this kernel
            //     vector<KernelMemoryAccessInfo> loc;
            //     for (int i = 0; i < query.args_size(); i++) {
            //         if (query.args(i).length() % sizeof(uint64_t) == 0) {
            //             for (int j = 0; j <  query.args(i).length() / sizeof(uint64_t); j++) {
            //                 // std::cout << " check arg_" << i << " val_" << j << " " << ((uint64_t*) args[i])[j];
            //                 if (mem_manager_.is_model_address(active_func_, (void*) ((uint64_t*) args[i])[j])) {
            //                     loc.insert({i, j, true});
            //                 }
            //                 else if (mem_manager_.valid_memory_address(active_func_, (void*) ((uint64_t*) args[i])[j])) {
            //                     loc.insert({i, j, false});
            //                 }
            //             }
            //         }
            //     }
            //     kernel_model_access_loc()[ptr] = loc;
            // }

            // for (auto &info : kernel_model_access_loc()[ptr]) {
            //     // std::cout << " arg_" << pair.first << "[" << pair.second << "] = " << ((uint64_t*) args[pair.first])[pair.second];
            //     if (info.is_model_access_) {
            //         mem_manager_.try_insert_model_access((void*) ((uint64_t*) args[info.arg_idx_])[info.offset_]);
            //     }
            //     ((uint64_t*) args[info.arg_idx_])[info.offset_] = (uint64_t) mem_manager_.get_cuda_addr(active_func_, (void*) ((uint64_t*) args[info.arg_idx_])[info.offset_]);
            //     // std::cout << " / " << ((uint64_t*) args[pair.first])[pair.second];
            // }

            // // std::cout << std::endl;

            size_t sharedMem = query.shared_mem();
            cudaStream_t stream = get_stream(query.stream());

            cudaCheck(cudaLaunchKernel(ptr, gridDim, blockDim, args, sharedMem, stream));
            // log_->debug("Async: cudaLaunchKernel {}", query.function());
        }
        else{
            std::cout << "Server " << id_  << " -- Not found func addr " << query.function() << std::endl;
        }
    }

private:
    template <typename RespType>
    inline void send_response(RespType& resp, int client_id) {
        string resp_serialized;
        resp.SerializeToString(&resp_serialized);
        kZmqUtil->send_string(resp_serialized, &pushers_[get_client_addr(client_id)]);
    }

    template <typename DescType>
    inline void insert_desc(const string& desc_id, DescType &desc, map<string, DescType>& desc_map) {
        if (desc_map.find(desc_id) == desc_map.end()) {
            desc_map[desc_id] = desc;
        }
        else {
            std::cout << "Server " << id_  << " -- Error: repeated descriptor id " << desc_id << std::endl;
        }
    }

    template <typename DescType>
    inline DescType get_desc(const string& desc_id, map<string, DescType>& desc_map) {
        if (desc_map.find(desc_id) != desc_map.end()) {
            return desc_map[desc_id];
        }
        else{
            std::cout << "Server " << id_  << " -- Error: cannot find descriptor " << desc_id << std::endl;
            throw std::runtime_error("Error: cannot find descriptor " + desc_id); // TODO: catch error and return to client
        }
    }

    inline cudnnHandle_t get_cudnn_handle(const string& handle_msg) {
        return cudnn_handle_;
    }
    
    inline cublasHandle_t get_cublas_handle(const string& handle_msg) {
        return cublas_handle_;
    }

    inline cudaStream_t create_stream(unsigned flags, int priority) {
        // TODO stream pool
        return secondary_stream_;
    }

    inline cudaStream_t get_stream(const string& stream_msg) {
        cudaStream_t input_stream;
        memcpy(&input_stream, &stream_msg[0], stream_msg.length());
        // return input_stream == 0 ? default_stream_ : input_stream;
        return input_stream == 0 ? default_stream_ : secondary_stream_;
    }

    // inline vector<pair<int, int>> get_model_access_in_kernel_launch(string& function, void** args, vector)

private:
    zmq::socket_t srv_pull_handler_;
    Pusher pushers_;
    vector<zmq::pollitem_t> pollitems_;

    cudaStream_t default_stream_;
    cudaStream_t secondary_stream_; // a general stream for non-blocking operations

    string active_func_;
    vector<AyncMessage> stream_async_msg_;

    manager::MemoryManager mem_manager_;

    int load_model_flag_;
    std::chrono::time_point<std::chrono::system_clock> req_start_;

    // map<const void*, ordered_set<pair<int, int>>> kernel_model_access_loc_;

    // cache cudnn cublas handles
    cudnnHandle_t cudnn_handle_;
    cublasHandle_t cublas_handle_;

    // collections of local descriptors
    map<string, cudnnTensorDescriptor_t> tensor_desc_map_;
    map<string, cudnnConvolutionDescriptor_t> conv_desc_map_;
    map<string, cudnnFilterDescriptor_t> filter_desc_map_;

    logger log_;
    int id_;

    int num_gpus_;

    ExecutorSyncQueue executor_sync_queue_;
    ExecutorRespBlockQueue executor_resp_block_queue_;
};

}
#endif  // INCLUDE_CUDA_SERVER_HPP_

