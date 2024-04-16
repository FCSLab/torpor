#ifndef INCLUDE_MEM_MANAGER_HPP_
#define  INCLUDE_MEM_MANAGER_HPP_

#include "cuda_common.hpp"
#include "utils.hpp"
#include "safe_ptr.hpp"
#include <cstdlib>
#include "model_repo.hpp"
#include "block_manager.hpp"
#include <cmath>

#define memoryCheck(ans) { memoryAssert((ans), __FILE__, __LINE__); }
inline void memoryAssert(bool suc, const char *file, int line) {
   if (!suc) {
      fprintf(stderr,"Memory access error: %s %d\n", file, line);
   }
}

namespace server {
namespace manager {

const size_t default_buffer_size = 2 * 1024 * 1024;
const int default_num_io_thread = 4;

const int PipeSignal_Exit = 0;
const int PipeSignal_Transfer = 1;
// const int PipeSignal_TransferP2P = 2;

const int ParamReadiness_Host = 0;
const int ParamReadiness_Pinned = 1;
const int ParamReadiness_CUDA = 2;

using ParamLocMap = sf::safe_ptr<map<void*, pair<int, void*>>>; 


class MemoryManager {
public:
    MemoryManager(size_t budget): track_memory_flag_(std::make_pair(false, "")) {
        sig_queue_ = create_blocking_queue<pair<int, string>>();
        sync_pipe_queue_ = create_blocking_queue<int>();

        // extra budget for cudaMalloc
        size_t total_budget = 5;
        total_budget = total_budget * 1024 * 1024 * 1024 + budget;

        auto block_manager_str = get_env_var("BLOCK_MANAGER");
        if (block_manager_str.size() > 0 && block_manager_str == "Fixed") {
            block_manager_ = std::make_shared<FixedSizeBlockManager>(total_budget);
        }
        else {
            block_manager_ = std::make_shared<BuddyBasedBlockManager>(total_budget);
        }
        
    }
    /**
    * Initialization
    */
    void set_logger(logger log) {
        log_ = log;
    }

    void set_server_id(int server_id) {
        server_id_ = server_id;
        block_manager_->set_server_id(server_id);
    }

    void set_track_model_memory(string &func) {
        track_memory_flag_ = std::make_pair(true, func);
        model_repo_.start_model_load(func);
        log_->info("Track memory for {}", func);
    }
    
    void unset_track_model_memory(string &func) {
        if (track_memory_flag_.first) {
            model_repo_.complete_model_load(func);
        }
        track_memory_flag_ = std::make_pair(false, "");
        block_manager_->finish_block_alloc();
        log_->info("Untrack memory for {}", func);
    }

    /**
    * cuda memory APIs
    */
    void* cudaMallocBackend(size_t size, bool load_model_flag, string& function){
        void* ptr;

        block_manager_->acquire_block(&ptr, size, true);

        model_repo_.malloc_block_for_model(ptr, size, load_model_flag, function);
        model_repo_.device_model_info_map_[server_id_][function].block_virtual_to_cuda_[ptr] = {true, false, nullptr, ptr};

        return ptr;
    }

    // In general users do not explicitly call cudaFree
    void cudaFreeBackend(void* ptr, string& active_func){
        if (model_repo_.device_model_info_map_[server_id_][active_func].block_virtual_to_cuda_.find(ptr) != model_repo_.device_model_info_map_[server_id_][active_func].block_virtual_to_cuda_.end()) {
            if (model_repo_.model_host_info_map_[active_func].virtual_blocks_[ptr].is_model_) {
                std::cout << "Server " << server_id_  << " -- Unexpected behaviour: free model block" << std::endl;
            }
            else {
                if (model_repo_.device_model_info_map_[server_id_][active_func].block_virtual_to_cuda_[ptr].in_cuda_) {
                    block_manager_->evict_block(model_repo_.device_model_info_map_[server_id_][active_func].block_virtual_to_cuda_[ptr].cuda_addr_, model_repo_.model_host_info_map_[active_func].virtual_blocks_[ptr].size_);
                }

                model_repo_.free_block_for_model(ptr, active_func);
            }
        }
    }

    void cudaMemcpyBackendHtoD(void* dst, const string &payload, string& active_func){
       cudaCheck(cudaMemcpy(get_phy_address(dst, active_func), &payload[0], payload.length(), cudaMemcpyKind::cudaMemcpyHostToDevice));
    }

    void cudaMemcpyBackendDtoH(void* local_dst, void* src, size_t count, string& active_func){
        cudaCheck(cudaMemcpy(local_dst, get_phy_address(src, active_func), count, cudaMemcpyKind::cudaMemcpyDeviceToHost));
    }

    void cudaMemcpyBackendDtoD(void* dst, void* src, size_t count, string& active_func){
        cudaCheck(cudaMemcpy(get_phy_address(dst, active_func), get_phy_address(src, active_func), count, cudaMemcpyKind::cudaMemcpyDeviceToDevice));
    }

    void cudaMemcpyAsyncBackendHtoD(void* dst, const string &payload, cudaStream_t stream, bool load_model_flag,  string& function){ 
        auto size = payload.length();

        if (load_model_flag) {
            if (model_repo_.load_model_h2d_at_start(dst, payload, size, function)){
                model_repo_.device_model_info_map_[server_id_][function].param_virtual_to_cuda_[dst] = {nullptr, dst};
            }
            else{
                log_->warn("cudaMemcpy model fails to find malloc {}", (uint64_t) dst);
                std::cout << "Server " << server_id_  <<  " -- Warning: cudaMemcpy model fails to find malloc " << (uint64_t) dst << std::endl;
            }
        }
        else {
            // data transfer during execution, e.g., input tensor
            dst = get_phy_address(dst, function);
        }

        cudaCheck(cudaMemcpyAsync(dst, &payload[0], size, cudaMemcpyKind::cudaMemcpyHostToDevice, stream));
    }

    void cudaMemcpyAsyncBackendDtoH(void* local_dst, void* src, size_t count, cudaStream_t stream, string& active_func){
        cudaCheck(cudaMemcpyAsync(local_dst, get_phy_address(src, active_func), count, cudaMemcpyKind::cudaMemcpyDeviceToHost, stream));
    }

    void cudaMemcpyAsyncBackendDtoD(void* dst, void* src, size_t count, cudaStream_t stream, string& active_func){
        if (is_model_address(active_func, src)) {
            memoryCheck(try_insert_model_access(src));
            src = check_param_readiness(src, active_func);
        }
        else {
            src = get_phy_address(src, active_func);
        }

        dst = get_phy_address(dst, active_func);
        cudaCheck(cudaMemcpyAsync(dst, src, count, cudaMemcpyKind::cudaMemcpyDeviceToDevice, stream));
    }

    void cudaMemsetAsyncBackend(void* ptr, int value, size_t count, cudaStream_t stream, string& active_func){
        cudaCheck(cudaMemsetAsync(get_phy_address(ptr, active_func), value, count, stream));
    }

    /**
    * utils
    */
    void start_trans_thread(int gpu_count) {
        auto buffer_size_str = get_env_var("BUFFER_SIZE");
        size_t buffer_size = default_buffer_size;
        if (buffer_size_str.size() > 0) {
            buffer_size = std::stoi(buffer_size_str);
        }

        block_manager_->init_allocation();
        model_repo_.init_device_info(server_id_);
        std::cout << "Server " << server_id_ << " -- Set transmission buffer size to " << buffer_size << std::endl;
        trans_thread_ = std::thread(&MemoryManager::transfer_model, this, buffer_size, gpu_count);

        num_io_threads_ = default_num_io_thread;
        auto num_io_thread_str = get_env_var("IO_THREAD_NUM");
        if (num_io_thread_str.size() > 0) {
            num_io_threads_ = std::stoi(num_io_thread_str);
        }
        std::cout << "Server " << server_id_ << " -- Set I/O thread num to " << num_io_threads_ << std::endl;

        io_queue_ = create_blocking_queue<DataCpyInfo>();
        sync_io_queue_ = create_blocking_queue<int>();

        for (int i = 0; i < num_io_threads_; i++) {
            io_threads_.emplace_back(std::thread(&MemoryManager::pinned_io_thread, this, i));
        }
    }

    bool check_model_alloc_status(string &function) {
        if (model_repo_.model_host_info_map_.find(function) != model_repo_.model_host_info_map_.end()) {
            if (model_repo_.device_model_info_map_[server_id_].find(function) == model_repo_.device_model_info_map_[server_id_].end()) {
                return false;
            }
            for (auto &info : model_repo_.model_host_info_map_[function].virtual_blocks_) {
                
                if (model_repo_.device_model_info_map_[server_id_][function].block_virtual_to_cuda_.find(info.first) == model_repo_.device_model_info_map_[server_id_][function].block_virtual_to_cuda_.end()
                        || !model_repo_.device_model_info_map_[server_id_][function].block_virtual_to_cuda_[info.first].in_cuda_) {
                    return false;
                }
            }
            return true;
        }
        return false;
    }


    void sig_trans_thread(string &function, int src_gpu = -1) {
        if (src_gpu >= 0 && (model_repo_.device_model_info_map_[src_gpu].find(function) == model_repo_.device_model_info_map_[src_gpu].end()
                        || model_repo_.device_model_info_map_[src_gpu][function].status_ != ModelStatus_Full)) {
            std::cout << "Server " << server_id_ << " -- warning: transfer unhealthy model " << function << " status " << model_repo_.device_model_info_map_[src_gpu][function].status_ << " from GPU " << src_gpu << ", using host swap" << std::endl;
            // std::cout << "Server " << server_id_ << " -- num " << model_repo_.model_active_device_map_[function].size() << ": ";
            // for (auto &g : model_repo_.model_active_device_map_[function])
            //     std::cout << g << " | " << model_repo_.device_model_info_map_[g][function].status_ << ", ";
            // std::cout << std::endl;
            
            // return;
            src_gpu = -1;
        }
        check_and_malloc_model(function, src_gpu < 0);
        sig_queue_->enqueue(std::make_pair(PipeSignal_Transfer + 1 + src_gpu, function));
    }

    void destroy_trans_thread() {
        sig_queue_->enqueue(std::make_pair(PipeSignal_Exit, ""));
        trans_thread_.join();
    }

    void init_model_readiness(string &function, int flag = ParamReadiness_CUDA) {
        for (auto &info : model_repo_.model_host_info_map_[function].model_param_info_) {
            model_param_readiness_[function]->emplace(info.first, std::make_pair(flag, info.first));
        }  
    }

    void evict_model(string &function, bool free_memory = false) {
        if (model_param_readiness_.find(function) == model_param_readiness_.end()) {
            std::cout << "Server " << server_id_ << " -- warning: try to evict unregonized model " << function << std::endl;
            return;
        }
        auto evict_start = std::chrono::system_clock::now();

        for (auto &info : model_repo_.model_host_info_map_[function].virtual_blocks_) {
            if (model_repo_.device_model_info_map_[server_id_][function].block_virtual_to_cuda_[info.first].in_cuda_) {
                model_repo_.device_model_info_map_[server_id_][function].block_virtual_to_cuda_[info.first].in_cuda_ = false;
                block_manager_->evict_block(model_repo_.device_model_info_map_[server_id_][function].block_virtual_to_cuda_[info.first].cuda_addr_, info.second.size_, free_memory);
                if (info.second.is_model_)
                    reset_model_readiness(function, info.first, model_repo_.device_model_info_map_[server_id_][function].block_virtual_to_cuda_[info.first].in_pin_ ? ParamReadiness_Pinned : ParamReadiness_Host);
            }
        }

        block_manager_->finish_release_block();
        std::cout << "Server " << server_id_ << " -- evicted model " << function << std::endl;
        log_->info("Evict model {} elasped {}", function, std::chrono::duration_cast<std::chrono::microseconds> (std::chrono::system_clock::now() - evict_start).count());   
    }

    inline void* get_cuda_addr(string& function, void* ptr, bool search_amid = true) {
        if (model_repo_.device_model_info_map_[server_id_].find(function) != model_repo_.device_model_info_map_[server_id_].end()) {
            if (model_repo_.model_host_info_map_[function].model_param_info_.find(ptr) != model_repo_.model_host_info_map_[function].model_param_info_.end()) {
                return check_param_readiness(ptr, function);
            }
            else {
                return get_phy_address(ptr, function, search_amid);
            }
        }
        return ptr;
    }

    inline bool try_insert_model_access(void* dev_ptr) {
        if (track_memory_flag_.first){
            if (model_repo_.model_host_info_map_[track_memory_flag_.second].model_param_info_.find(dev_ptr) != model_repo_.model_host_info_map_[track_memory_flag_.second].model_param_info_.end()) {
                model_repo_.check_add_model_access_order(track_memory_flag_.second, dev_ptr);
            }
            else {
                std::cout << "Server " << server_id_  << " -- Model " << track_memory_flag_.second << " parameter ptr not found " << (uint64_t) dev_ptr << std::endl;
                return false;
            }
        }
        return true;
    }

    inline bool is_model_address(string& function, void* addr) {
        if (model_repo_.model_host_info_map_.find(function) != model_repo_.model_host_info_map_.end()) {
            return model_repo_.model_host_info_map_[function].model_param_info_.find(addr) != model_repo_.model_host_info_map_[function].model_param_info_.end();
        }
        return false;
    }

    inline bool valid_memory_address(string& function, void* addr) {
        if (model_repo_.model_host_info_map_.find(function) != model_repo_.model_host_info_map_.end()) {
            model_repo_.find_block_ephe_ptr_mapping(function, addr);
        }
        return false;
    }

private:
    inline void reset_model_readiness(string &function, void* block_addr, int flag = ParamReadiness_Host) {

        if (model_repo_.model_host_info_map_[function].virtual_blocks_.find(block_addr) != model_repo_.model_host_info_map_[function].virtual_blocks_.end()) {
            for (auto &info : model_repo_.model_host_info_map_[function].model_block_info_[block_addr].virtual_param_ptrs_) {
                if (model_param_readiness_[function]->find(info) != model_param_readiness_[function]->end())
                    model_param_readiness_[function]->at(info) = std::make_pair(flag, nullptr);
            }
        }
    }

    void pinned_io_thread (int io_thread_id) noexcept {
        while (true) {
            auto info = io_queue_->dequeue();
            if (info.size_ < 0) {
                return;
            }
            if (info.src_ == nullptr || info.dst_ == nullptr) {
                std::cout << "Server " << server_id_ << " -- pinned io thread " << io_thread_id << " src " << (uint64_t) info.src_ << " dst " << (uint64_t) info.dst_ << " size " << info.size_ << std::endl;
                if (model_param_readiness_[info.function_]->find(info.virtual_ptr_) != model_param_readiness_[info.function_]->end())
                    model_param_readiness_[info.function_]->at(info.virtual_ptr_) = std::make_pair(ParamReadiness_CUDA, nullptr);
            }
            else {
                memcpy(info.dst_, info.src_, info.size_);
                if (model_param_readiness_[info.function_]->find(info.virtual_ptr_) != model_param_readiness_[info.function_]->end())
                    model_param_readiness_[info.function_]->at(info.virtual_ptr_) = std::make_pair(ParamReadiness_Pinned, nullptr);
            }
            sync_io_queue_->enqueue(1);
        }
    }

    void transfer_model(size_t buffer_size, int gpu_count) noexcept {
        for (int i = 0; i < gpu_count; i++) {
            cudaCheck(cudaSetDevice(i));
            cudaStream_t stream;
            cudaEvent_t event;
            cudaCheck(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, -1));
            cudaCheck(cudaEventCreate(&event));
            trans_stream_.push_back(stream);
            trans_event_.push_back(event);
        }

        cudaCheck(cudaSetDevice(server_id_));

        while (true) {
            auto signal = sig_queue_->dequeue();
            if (signal.first >= PipeSignal_Transfer) {
                // start transmission

                int src_gpu_id = signal.first - PipeSignal_Transfer - 1;
                std::cout << "Server " << server_id_ << " -- Start transfering model " << signal.second << " from GPU " << src_gpu_id << std::endl;
                log_->info("Start transfering model {} from GPU {}", signal.second, src_gpu_id);
                auto start_t = std::chrono::system_clock::now();
                int param_count_host_to_pin = 0;
                if (src_gpu_id < 0) {
                    // local model to pinned memory
                    for (auto &ptr: model_repo_.model_access_order_map_[signal.second]){
                        if (model_param_readiness_[signal.second]->find(ptr) == model_param_readiness_[signal.second]->end() 
                                || model_param_readiness_[signal.second]->at(ptr).first >= ParamReadiness_Pinned) 
                            continue;
                        io_queue_->enqueue({signal.second, ptr, 
                                model_repo_.device_model_info_map_[server_id_][signal.second].param_virtual_to_cuda_[ptr].pinned_addr_, 
                                model_repo_.model_host_info_map_[signal.second].model_param_info_[ptr].host_addr_,
                                model_repo_.model_host_info_map_[signal.second].model_param_info_[ptr].size_});
                        // memcpy(model_repo_.device_model_info_map_[server_id_][signal.second].param_virtual_to_cuda_[ptr].pinned_addr_, 
                        //         model_repo_.model_host_info_map_[signal.second].model_param_info_[ptr].host_addr_,
                        //         model_repo_.model_host_info_map_[signal.second].model_param_info_[ptr].size_);
                        // model_param_readiness_[signal.second]->at(ptr) = std::make_pair(ParamReadiness_Pinned, nullptr);
                        param_count_host_to_pin ++;
                    }
                }
                auto to_pin_t = std::chrono::system_clock::now();

                // std::cout << "Server " << server_id_ << " -- pinned model " << signal.second << std::endl;

                set<void*> issued_ptrs;
                size_t cur_size = 0;
                int wait_count = 0;
                for (auto &ptr: model_repo_.model_access_order_map_[signal.second]){
                    if (model_param_readiness_[signal.second]->find(ptr) == model_param_readiness_[signal.second]->end() 
                            || model_param_readiness_[signal.second]->at(ptr).first == ParamReadiness_CUDA) 
                        continue;

                    if (src_gpu_id < 0) {
                        // wait for the param to be pinned
                        while (model_param_readiness_[signal.second]->at(ptr).first < ParamReadiness_Pinned) {
                            sync_io_queue_->dequeue();
                            wait_count++;
                        }
                        cudaMemcpyAsync(model_repo_.device_model_info_map_[server_id_][signal.second].param_virtual_to_cuda_[ptr].cuda_addr_, 
                                        model_repo_.device_model_info_map_[server_id_][signal.second].param_virtual_to_cuda_[ptr].pinned_addr_, 
                                        model_repo_.model_host_info_map_[signal.second].model_param_info_[ptr].size_, cudaMemcpyHostToDevice, trans_stream_[server_id_]);
                    }
                    else {
                        cudaMemcpyPeerAsync(model_repo_.device_model_info_map_[server_id_][signal.second].param_virtual_to_cuda_[ptr].cuda_addr_, server_id_, 
                                            model_repo_.device_model_info_map_[src_gpu_id][signal.second].param_virtual_to_cuda_[ptr].cuda_addr_, src_gpu_id,
                                            model_repo_.model_host_info_map_[signal.second].model_param_info_[ptr].size_, trans_stream_[src_gpu_id]);
                    }

                    cur_size += model_repo_.model_host_info_map_[signal.second].model_param_info_[ptr].size_;
                    issued_ptrs.insert(ptr);
                    if (cur_size < buffer_size) {
                        continue;
                    }
                    sync_with_compute(issued_ptrs, signal.second, src_gpu_id < 0 ? server_id_ : src_gpu_id);
                    cur_size = 0;
                }
                if (issued_ptrs.size() > 0){
                    sync_with_compute(issued_ptrs, signal.second, src_gpu_id < 0 ? server_id_ : src_gpu_id);
                }

                // std::cout << "Server " << server_id_ << " -- done transfer model " << signal.second << std::endl;

                auto end_t = std::chrono::system_clock::now();
                if (src_gpu_id < 0) {
                    // TODO clear sync_io_queue
                    for (auto &info: model_repo_.model_host_info_map_[signal.second].model_block_info_){
                        if (model_repo_.device_model_info_map_[server_id_][signal.second].block_virtual_to_cuda_[info.first].in_pin_) {
                            free_pinend_memory(model_repo_.device_model_info_map_[server_id_][signal.second].block_virtual_to_cuda_[info.first].pinned_addr_, 
                                            model_repo_.model_host_info_map_[signal.second].virtual_blocks_[info.first].size_);
                            model_repo_.device_model_info_map_[server_id_][signal.second].block_virtual_to_cuda_[info.first].in_pin_ = false;
                        }
                    }
                }

                std::cout << "Server " << server_id_ << " -- Transfer model " << signal.second << " h2p count " << param_count_host_to_pin
                                << " start_t " <<  std::chrono::duration_cast<std::chrono::microseconds>(start_t.time_since_epoch()).count()
                                << " pinned elasped " << std::chrono::duration_cast<std::chrono::microseconds> (to_pin_t - start_t).count()
                                << " cuda elasped " << std::chrono::duration_cast<std::chrono::microseconds> (end_t - to_pin_t).count() << std::endl;
                log_->info("Transfer model {} from {} wait count {} pinned elasped {} overall elasped {}", signal.second, src_gpu_id, wait_count, 
                            std::chrono::duration_cast<std::chrono::microseconds> (to_pin_t - start_t).count(),
                            std::chrono::duration_cast<std::chrono::microseconds> (end_t - start_t).count());
            }
            else if (signal.first == PipeSignal_Exit) {
                // exit
                return;
            }
        }
    }


    inline void sync_with_compute(set<void*> &issued_ptrs, string &function, int gpu_id) {
        cudaEventRecord(trans_event_[gpu_id], trans_stream_[gpu_id]);
        cudaEventSynchronize(trans_event_[gpu_id]);
        for (auto ptr: issued_ptrs) {
            model_param_readiness_[function]->at(ptr) = std::make_pair(ParamReadiness_CUDA, model_repo_.device_model_info_map_[server_id_][function].param_virtual_to_cuda_[ptr].cuda_addr_);
        }
        sync_pipe_queue_->enqueue(1);
        issued_ptrs.clear();
    }

    void check_and_malloc_model(string &function, bool alloc_pinned = true) {
        if (model_repo_.model_host_info_map_.find(function) != model_repo_.model_host_info_map_.end()) {
            auto check_start = std::chrono::system_clock::now();
            if (model_repo_.device_model_info_map_[server_id_].find(function) == model_repo_.device_model_info_map_[server_id_].end()) {
                model_repo_.init_device_model_info(server_id_, function);
            }
            if (model_param_readiness_.find(function) == model_param_readiness_.end()) {
                init_model_readiness(function, ParamReadiness_Host);
            }

            for (auto &info : model_repo_.model_host_info_map_[function].virtual_blocks_) {
                // std::cout << "Server " << server_id_ << " -- check_and_malloc_model " << function << " block " << (uint64_t) info.first << " size " << info.second.size_ << std::endl;
                if (model_repo_.device_model_info_map_[server_id_][function].block_virtual_to_cuda_.find(info.first) == model_repo_.device_model_info_map_[server_id_][function].block_virtual_to_cuda_.end()
                        || !model_repo_.device_model_info_map_[server_id_][function].block_virtual_to_cuda_[info.first].in_cuda_) {
                    block_manager_->acquire_block(&model_repo_.device_model_info_map_[server_id_][function].block_virtual_to_cuda_[info.first].cuda_addr_, info.second.size_);
                    model_repo_.device_model_info_map_[server_id_][function].block_virtual_to_cuda_[info.first].in_cuda_ = true;

                    // std::cout << "Server " << server_id_ << " -- in malloc " << (uint64_t) info.first << " is model " << info.second.is_model_ << std::endl;

                    // update physical data ptr of associated params
                    if (info.second.is_model_) {

                        if (alloc_pinned && !model_repo_.device_model_info_map_[server_id_][function].block_virtual_to_cuda_[info.first].in_pin_) {
                            acquire_pinned_memory(&model_repo_.device_model_info_map_[server_id_][function].block_virtual_to_cuda_[info.first].pinned_addr_, info.second.size_);
                        }

                        for (auto &data_ptr : model_repo_.model_host_info_map_[function].model_block_info_[info.first].virtual_param_ptrs_) {
                            auto phy_ptr = (uint64_t) data_ptr - (uint64_t) info.first + (uint64_t) model_repo_.device_model_info_map_[server_id_][function].block_virtual_to_cuda_[info.first].cuda_addr_;
                            model_repo_.device_model_info_map_[server_id_][function].param_virtual_to_cuda_[data_ptr].cuda_addr_ = (void*) phy_ptr;
                            
                            if (alloc_pinned && !model_repo_.device_model_info_map_[server_id_][function].block_virtual_to_cuda_[info.first].in_pin_) {
                                auto pinned_ptr = (uint64_t) data_ptr - (uint64_t) info.first + (uint64_t) model_repo_.device_model_info_map_[server_id_][function].block_virtual_to_cuda_[info.first].pinned_addr_;
                                model_repo_.device_model_info_map_[server_id_][function].param_virtual_to_cuda_[data_ptr].pinned_addr_ = (void*) pinned_ptr;
                            }

                            // model_param_readiness_[function]->at(data_ptr) = std::make_pair(ParamReadiness_Host, nullptr);
                        }
                        if (alloc_pinned && !model_repo_.device_model_info_map_[server_id_][function].block_virtual_to_cuda_[info.first].in_pin_) {
                            model_repo_.device_model_info_map_[server_id_][function].block_virtual_to_cuda_[info.first].in_pin_ = true;
                        }
                    }
                }
            }
            active_model_params_.clear();
            std::copy(model_repo_.model_access_order_map_[function].begin(), model_repo_.model_access_order_map_[function].end(), std::inserter(active_model_params_, active_model_params_.end()));

            block_manager_->finish_block_alloc();

            auto elasped = std::chrono::duration_cast<std::chrono::microseconds> (std::chrono::system_clock::now() - check_start).count();
            log_->info("CheckMalloc model {} elasped {}", function, elasped);
            std::cout << "Server " << server_id_ << " -- CheckMalloc model " << function << " elasped " << elasped << std::endl;
        }
        else {
            std::cout << "Server " << server_id_ << " -- Model " << function << " not found in model repo" << std::endl;
            log_->warn("Model {} not found in model repo", function);
        }
    }


    inline void acquire_pinned_memory(void** ptr, size_t size) {
        if (pinned_memory_blocks_[size].size() > 0) {
            *ptr = pinned_memory_blocks_[size].back();
            pinned_memory_blocks_[size].pop_back();
        }
        else {
            cudaCheck(cudaMallocHost(ptr, size));
            all_pinned_blocks_.insert(*ptr);
        }
    }

    inline void free_pinend_memory(void* ptr, size_t size) {
        if (all_pinned_blocks_.find(ptr) != all_pinned_blocks_.end()) {
            pinned_memory_blocks_[size].push_back(ptr);
        }
    }

    inline void* check_param_readiness(void* ptr, string &function) {
        if (model_param_readiness_.find(function) != model_param_readiness_.end()) {
            if (active_model_params_.find(ptr) != active_model_params_.end() && model_param_readiness_[function]->find(ptr) != model_param_readiness_[function]->end()) {
                log_->debug("Check accessed paramter model {} ptr {}", function, (uint64_t) ptr);
                while (model_param_readiness_[function]->at(ptr).first != ParamReadiness_CUDA) sync_pipe_queue_->dequeue();
                return model_param_readiness_[function]->at(ptr).second;   
            }
            else {
                log_->debug("Skip unaccessed paramter model {} ptr {}", function, (uint64_t) ptr);
                return model_repo_.device_model_info_map_[server_id_][function].param_virtual_to_cuda_[ptr].cuda_addr_;
            }
        }
        return ptr;
    }

    inline void* get_phy_address(void* ptr, string& function, bool search_amid = true) {
        // if ((uint64_t) ptr == 0) return ptr;

        if (model_repo_.model_host_info_map_.find(function) != model_repo_.model_host_info_map_.end()) {

            if (model_repo_.model_host_info_map_[function].virtual_data_block_mapping_.find(ptr) == model_repo_.model_host_info_map_[function].virtual_data_block_mapping_.end()) {
                if (!search_amid || !valid_memory_address(function, ptr)) {
                    // log_->warn("func {} ptr {} is not a memory access", function, (uint64_t) ptr);
                    // std::cout << "Server " << server_id_  << " -- Warning: func " << function << " ptr " << (uint64_t) ptr << " is not a memory access" << std::endl;
                    return ptr;
                }
            }
            if (model_repo_.device_model_info_map_[server_id_][function].block_virtual_to_cuda_[model_repo_.model_host_info_map_[function].virtual_data_block_mapping_[ptr]].in_cuda_ 
                    && model_repo_.device_model_info_map_[server_id_][function].block_virtual_to_cuda_[model_repo_.model_host_info_map_[function].virtual_data_block_mapping_[ptr]].cuda_addr_ != nullptr) {
                return (void*) ((uint64_t) ptr - (uint64_t) model_repo_.model_host_info_map_[function].virtual_data_block_mapping_[ptr] 
                            + (uint64_t) model_repo_.device_model_info_map_[server_id_][function].block_virtual_to_cuda_[model_repo_.model_host_info_map_[function].virtual_data_block_mapping_[ptr]].cuda_addr_);
            }
            else {
                std::cout << "Server " << server_id_  << " -- Warning: func " << function << " block " << (uint64_t) model_repo_.model_host_info_map_[function].virtual_data_block_mapping_[ptr] 
                                    << " (" << (uint64_t) model_repo_.device_model_info_map_[server_id_][function].block_virtual_to_cuda_[model_repo_.model_host_info_map_[function].virtual_data_block_mapping_[ptr]].cuda_addr_
                                    << ") not in cuda. ptr " << (uint64_t) ptr << std::endl;
                return ptr;
            }
        }
        std::cout << "Server " << server_id_  << " -- Warning: func " << function << " ptr " << (uint64_t) ptr << " not found" << std::endl;
        return ptr;
    }

private:
    logger log_;
    int server_id_;

    int num_io_threads_;

    pair<bool, string> track_memory_flag_;

    blocking_queue<pair<int, string>> sig_queue_;
    blocking_queue<int> sync_pipe_queue_;
    blocking_queue<DataCpyInfo> io_queue_;
    blocking_queue<int> sync_io_queue_;

    std::thread trans_thread_;
    vector<std::thread> io_threads_;

    vector<cudaStream_t> trans_stream_;
    vector<cudaEvent_t> trans_event_;

    ModelRepo& model_repo_ = ModelRepo::getInstance();
    map<string, ParamLocMap> model_param_readiness_;
    set<void*> active_model_params_;

    map<size_t, vector<void*>> pinned_memory_blocks_;
    set<void*> all_pinned_blocks_;

    std::shared_ptr<BlockManager> block_manager_;

};

}
}

#endif  // INCLUDE_MEM_MANAGER_HPP_
