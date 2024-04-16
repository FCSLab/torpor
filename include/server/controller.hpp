#ifndef INCLUDE_CONTROLLER_HPP_
#define  INCLUDE_CONTROLLER_HPP_

#include "cuda_common.hpp"
#include <chrono>
#include <algorithm>
#include "utils.hpp"
#include <list>
#include "socket_helper.hpp"
#include "cuda_server.hpp"
#include "model_repo.hpp"
#include "evictor.hpp"
#include "peer_interface.hpp"
#include "peer_tcp.hpp"

namespace server {

const int ExecutorStatus_Avail = -1;
const int ExecutorStatus_RunLocal = 0;
const int ExecutorStatus_RunNvlink = 1;
const int ExecutorStatus_RunPcleLight = 2;
const int ExecutorStatus_RunPcleMedian = 3;
const int ExecutorStatus_RunPcleHeavy = 4;
const int ExecutorStatus_RunColdStart = 5;

inline int model_level_to_exec_status(int model_level) {
    if (model_level >= ModelLevel_Light && model_level <= ModelLevel_Heavy) {
        return model_level + ExecutorStatus_RunPcleLight - ModelLevel_Light;
    }
    else {
        return ExecutorStatus_RunPcleMedian;
    }
}

struct ExecutorStatus {
    int status_ = ExecutorStatus_Avail;
    string function_ = "";
};

struct ScheduleDecision {
    int gpu_idx_;
    int src_swap_gpu_;
    int status_;
};

const int src_gpu_for_test = 3;

const size_t default_gpu_mem_limit = 25;
const string default_schedule_policy = "IA"; // interference-aware (IA) or random (RAND)
const string default_eviction_policy = "IA"; // interference-aware (IA) or LRU

class Controller {
public:
    Controller(logger log, zmq::context_t *context,unsigned int signal_port = 0): 
            log_(log),
            socket_(zmq::socket_t(*context, ZMQ_DEALER)),
            sig_socket_(zmq::socket_t(*context, ZMQ_ROUTER)) {
        cudaCheck(cudaGetDeviceCount(&gpu_count_));

        // init cudnn handle
        cudnnHandle_t init_cudnn_handle;
        cudnnCheck(cudnnCreate(&init_cudnn_handle));

        for (int i = 0; i < gpu_count_; i++) {
            cudaCheck(cudaSetDevice(i));

            for (int j = 0; j < gpu_count_; j++) {
                if (i == j) continue;
                int is_able;
                cudaDeviceCanAccessPeer(&is_able, i, j);
                if (is_able) {
                    auto err = cudaDeviceEnablePeerAccess(j, 0);
                    std::cout << "Controller -- enable peer access cur " << i << " peer " << j << " res " << err << std::endl;
                }
            }
        }
        
        if(signal_port != 0){
            string addr = bind_signal_addr(signal_port);
            sig_socket_.bind(bind_signal_addr(signal_port));

            address_to_socket_map_.insert(make_pair(addr,&sig_socket_));
        }else{
            string addr = bind_signal_addr();
            sig_socket_.bind(bind_signal_addr());
            socket_.connect("tcp://localhost:90");

            address_to_socket_map_.insert(make_pair(addr,&sig_socket_));
        }

        //sig_socket_.bind(bind_signal_addr());

        pollitems_ = {
            {static_cast<void*>(sig_socket_), 0, ZMQ_POLLIN, 0} ,
            {static_cast<void*>(socket_), 0, ZMQ_POLLIN, 0} ,
        };
        
        auto gpu_mem_limit_str = get_env_var("MEM_LIMIT_IN_GB");
        size_t gpu_mem_limit = default_gpu_mem_limit;
        if (gpu_mem_limit_str.size() > 0) {
            int limit = std::stoi(gpu_mem_limit_str);
            if (limit > 0 && limit < default_gpu_mem_limit) {
                gpu_mem_limit = limit;
            }
        }

        std::cout << "Set GPU memory limit to " << gpu_mem_limit  << " GB"<< std::endl;
        gpu_mem_limit = gpu_mem_limit * 1024 * 1024 * 1024;

        auto schedule_policy_str = get_env_var("SCHEDULE_POLICY");
        if (schedule_policy_str.size() > 0 && schedule_policy_str == "RAND") {
            schedule_func_ = std::bind(&Controller::random_schedule, this, std::placeholders::_1);
        }
        else {
            schedule_func_ = std::bind(&Controller::interference_aware_schedule, this, std::placeholders::_1);
        }

        auto evict_policy_str = get_env_var("EVICT_POLICY");
        if (evict_policy_str.size() > 0 && evict_policy_str == "LRU") {
            evictor_ = std::make_shared<LRUEvictor>(gpu_count_, gpu_mem_limit);
        }
        else {
            evictor_ = std::make_shared<InterferenceAwareEvictor>(gpu_count_, gpu_mem_limit);
        }

        for (int i = 0; i < gpu_count_; i++){
            executor_vec_.push_back(std::make_shared<CUDAExecutor>(i, gpu_mem_limit, context));
            executor_status_.push_back({ExecutorStatus_Avail, ""});
        }

        // TODO import function pcie sharing and nvlink topology
        pcie_pairs_ = {1, 0, 3, 2};
        nvlink_topology_ = {{0, 1, 1, 2}, {1, 0, 2, 1}, {1, 2, 0, 2}, {2, 1, 2, 0}}; // -1: pcie only; 0: self; 1: slow link; 2: fast link

        log_->info("Start {} executor", gpu_count_);
    }

    void start() {
        for (int i = 0; i < gpu_count_; i++){
            thread_vec_.push_back(std::thread(&CUDAExecutor::start, executor_vec_[i]));
        }

        while (true) {
            kZmqUtil->poll(0, &pollitems_);
            // signal from router
            if (pollitems_[0].revents & ZMQ_POLLIN) {
                zmq::message_t address; 
                auto req_serialized = kZmqUtil->recv_string_with_peer(address,&sig_socket_);
                //auto req_serialized = kZmqUtil->recv_string(&sig_socket_);
                SignalRequest req;
                req.ParseFromString(req_serialized);
                auto func = req.function();

                std::cout << "Controller -- Receive signal " << req.type() << " for func " << func << std::endl;
                log_->info("Receive signal {} func {}", req.type(), func);

                if (req.type() == RequestType::ExecuteAfterLoad) {
                    model_level_map_[func] = ModelLevel_Median;
                    auto level = req.payload();
                    if (level.size() > 0) {
                        auto level_flag = std::stoi(level);
                        if (level_flag >= ModelLevel_Light && level_flag <= ModelLevel_Heavy) {
                            model_level_map_[func] = level_flag;
                        }
                        else {
                            std::cout << "Controller -- Unknown model level flag " << level_flag << std::endl;
                        }
                    }

                    auto gpu_idx = placement(func);                    
                    if (gpu_idx == -1) {
                        req_queue_.push({func, true});
                    }
                    else {
                        send_exec_req(func, gpu_idx, ExecutorStatus_RunColdStart);
                        send_signal_ack(address,AckType::OK, gpu_idx);
                    }
                }
                else if (req.type() == RequestType::Execute) {
                    auto hint = req.payload();
                    if (hint.size() > 0) {
                        // externel test
                        auto gpu_idx = std::stoi(hint);
                        // test p2p
                        // send_load_req(func, gpu_idx, gpu_idx == 0 && model_repo_.model_active_device_map_[func].find(src_gpu_for_test) != model_repo_.model_active_device_map_[func].end()? src_gpu_for_test : -1);
                        send_load_req(func, gpu_idx);
                        send_exec_req(func, gpu_idx, model_level_to_exec_status(model_level_map_[func]));
                        send_signal_ack(address,AckType::OK, gpu_idx);
                    }
                    else {
                        auto decision = schedule_func_(func);
                        if (decision.gpu_idx_ == -1) {
                            req_queue_.push({func, false});
                        }
                        else {
                            std::cout << "Controller -- Send func " << func << " exec req to " << decision.gpu_idx_ << " src gpu " << decision.src_swap_gpu_ << " status " << decision.status_ << std::endl;
                            send_load_req(func, decision.gpu_idx_, decision.src_swap_gpu_);
                            send_exec_req(func, decision.gpu_idx_, decision.status_);
                            send_signal_ack(address,AckType::OK, decision.gpu_idx_);
                        }
                    }
                }
                else if (req.type() == RequestType::ExecuteAfterP2PLoad) {
                    auto peer_address = req.payload();

                    auto decision = schedule_func_(func);
                    if (decision.gpu_idx_ == -1) {
                        std::cout << "" << std::endl;
                        req_queue_.push({func, false});
                    }
                    else {
                        std::cout << "Controller -- Send func " << func << " exec req to " << decision.gpu_idx_ << " src gpu " << decision.src_swap_gpu_ << " status " << decision.status_ << std::endl;
                        if(model_repo_.model_meta_data_info_map_.find(func)==model_repo_.model_meta_data_info_map_.end()){
                            zmq::socket_t* socket = h2h_tcp_.fetch_model(&socket_,peer_address,func,model_repo_);
                            //listen_to_socket(peer_address,socket);
                        }
                        send_load_req(func, decision.gpu_idx_, manager::REMOTE_ID);
                        send_exec_req(func, decision.gpu_idx_, decision.status_);
                        send_signal_ack(address,AckType::OK, decision.gpu_idx_);
                    }
                }
                else if (req.type() == RequestType::Unload) {
                    auto hint = req.payload();
                    if (hint.size() > 0) {
                        auto gpu_idx = std::stoi(hint);
                        if (model_repo_.model_active_device_map_[func].find(gpu_idx) != model_repo_.model_active_device_map_[func].end()) {
                            evictor_->force_evict(gpu_idx, func);
                            model_repo_.remove_model_info(func, gpu_idx);
                            send_to_executor(func, gpu_idx, ExecutorSignal_Unload);
                        }
                    }
                    else {
                        vector<int> gpu_list;
                        std::copy(model_repo_.model_active_device_map_[func].begin(), model_repo_.model_active_device_map_[func].end(), std::back_inserter(gpu_list));
                        for (auto &gpu_id : gpu_list) {
                            evictor_->force_evict(gpu_id, func);
                            model_repo_.remove_model_info(func, gpu_id);
                            send_to_executor(func, gpu_id, ExecutorSignal_Unload);
                        }
                    }
                    send_signal_ack(address,AckType::OK);
                }
                else if (req.type() == RequestType::Load) {
                    // test
                    for (int i = 0; i < gpu_count_; i++) {
                        send_load_req(func, i);
                        model_repo_.add_model_info(func, i);
                    }
                    send_signal_ack(address,AckType::OK);
                }else if(req.type() == RequestType::SendModel){
                    auto peer_address = req.payload();
                    if (peer_address.size() > 0) {
                        if(model_repo_.model_access_order_map_.find(func)!= model_repo_.model_access_order_map_.end()){
                            vector<void*> v_ptrs;
                            vector<int> param_size;
                            int param_count = 0;
                            for(auto ptr : model_repo_.model_access_order_map_[func]){
                                v_ptrs.push_back(ptr);
                                param_size.push_back(model_repo_.model_host_info_map_[func].model_param_info_[ptr].size_);
                                param_count++;
                            }
                            h2h_tcp_.send_model_meta_data(peer_address,
                                func,
                                v_ptrs,
                                param_size,
                                param_count
                                );
                            h2h_tcp_.send_model(peer_address,func,model_repo_);
                        }
                    }
                }
                else if (req.type() == RequestType::H2HfetchModel){
                    // test 
                    // vector<void*> v_ptrs;
                    // vector<int> param_size;
                    // int param_count = 1;

                    // h2h_tcp_.send_model_meta_data_resp(address,
                    //         &sig_socket_,
                    //         func,
                    //         v_ptrs,
                    //         param_size,
                    //         param_count
                    //         );
                    
                    // model_repo_.model_access_order_map_[func].push_back((void*)1);
                    // h2h_tcp_.send_model_resp(address,&sig_socket_,func,model_repo_);


                    if(model_repo_.model_access_order_map_.find(func)!= model_repo_.model_access_order_map_.end()){
                        vector<void*> v_ptrs;
                        vector<int> param_size;
                        int param_count = 0;
                        for(auto ptr : model_repo_.model_access_order_map_[func]){
                            v_ptrs.push_back(ptr);
                            param_size.push_back(model_repo_.model_host_info_map_[func].model_param_info_[ptr].size_);
                            param_count++;
                        }
                        h2h_tcp_.send_model_meta_data_resp(address,
                            &sig_socket_,
                            func,
                            v_ptrs,
                            param_size,
                            param_count
                            );
                        h2h_tcp_.send_model_resp(address,&sig_socket_,func,model_repo_);
                    }                    
                }
            }
            for(size_t i = 1; i < pollitems_.size(); ++i) {
                if (pollitems_[1].revents & ZMQ_POLLIN) {
                    std::cout<< "arrie here" <<std::endl;
                    //auto socket = reinterpret_cast<zmq::socket_t*>(pollitems_[1].socket);
                    std::cout<< "arrve here" <<std::endl;
                    auto req_serialized = kZmqUtil->recv_string(&socket_);
                    SignalRequest req;
                    req.ParseFromString(req_serialized);
                    auto func = req.function();

                    std::cout << "Controller -- Receive signal " << req.type() << " for func " << func << std::endl;
                    log_->info("Receive signal {} func {}", req.type(), func);
                    
                    if (req.type() == RequestType::H2HSendModelMetaData){
                        auto meta_data = req.meta_data();
                        int param_count = meta_data.param_count();
                        vector<void*> v_ptrs;

                        std::cout << "Controller -- ReceiveModelMetaData " << func << " param_count " << param_count << std::endl;
                        
                        // update model_access_order_map_
                        for(int i=0;i<param_count;i++){
                            void* v_ptr = (void*)meta_data.virtual_ptr(i);
                            v_ptrs.push_back(v_ptr);
                            model_repo_.check_add_model_access_order(func,v_ptr);
                        }
                        // init model_param_readiness
                        init_model_readiness_to_remote(func,v_ptrs);
                        model_repo_.model_meta_data_info_map_[func] = true;
                    }else if (req.type() == RequestType::H2HSendModelData){
                        auto param_data = req.param_data();

                        auto param = param_data.param();
                        auto index = param_data.index();
                        auto size = param_data.size();

                        std::cout << "Controller -- ReceiveModelParamData " << func << " index " << index << std::endl;

                        void* v_ptr = model_repo_.model_access_order_map_[func][index];
                        // update model_host_info_map_
                        model_repo_.load_model_param_to_host(func,v_ptr,param,size);

                        // update model_param_readiness
                        update_model_param_readiness_to_host(func,v_ptr);
                        sync_with_net();
                    }
                }
            }
            // handle signal from executors
            if (!controller_sync_queue->empty()) {
                auto executor_id = controller_sync_queue->front();
                controller_sync_queue->pop();

                if (executor_status_[executor_id].status_ == ExecutorStatus_RunColdStart) {
                    // update cache size if this is a cold start
                    model_repo_.add_model_info(executor_status_[executor_id].function_, executor_id);
                    check_eviction(executor_id, executor_status_[executor_id].function_);
                }
                else if (executor_status_[executor_id].status_ > ExecutorStatus_Avail && executor_status_[executor_id].status_ < ExecutorStatus_RunColdStart) {
                    // update model info
                    model_repo_.add_model_info(executor_status_[executor_id].function_, executor_id);
                }

                /* test swap via proactive eviction */
                // if (executor_id != src_gpu_for_test) {
                //     auto last_func = std::to_string(executor_status_[executor_id]);
                //     send_to_executor(last_func, executor_id, ExecutorSignal_Unload);
                // }

                unlock_src_func(executor_id);
                executor_status_[executor_id].status_ = ExecutorStatus_Avail;

                if (!req_queue_.empty()) {
                    auto func_sig_pair = req_queue_.front();
                    req_queue_.pop();
                    if (func_sig_pair.second) {
                        send_exec_req(func_sig_pair.first, executor_id, ExecutorStatus_RunColdStart);
                    }
                    else {
                        auto decision = schedule_func_(func_sig_pair.first);
                        if (decision.gpu_idx_ != executor_id) {
                            std::cout << "Controller -- warning only option " << executor_id << " not equal to schedule decision: " << decision.gpu_idx_ << " src gpu " << decision.src_swap_gpu_ << " status " << decision.status_ << std::endl;
                            send_load_req(func_sig_pair.first, executor_id, -1);
                            send_exec_req(func_sig_pair.first, executor_id, model_level_to_exec_status(model_level_map_[func_sig_pair.first]));
                        }
                        else {
                            std::cout << "Controller -- Send func " << func_sig_pair.first << " exec req to " << decision.gpu_idx_ << " src gpu " << decision.src_swap_gpu_ << " status " << decision.status_ << std::endl;
                            send_load_req(func_sig_pair.first, decision.gpu_idx_, decision.src_swap_gpu_);
                            send_exec_req(func_sig_pair.first, decision.gpu_idx_, decision.status_);
                        }
                    }
                    send_signal_ack(AckType::OK, executor_id);
                }
            }
        }
    }

private:
    /* trivial scheduling */
    int naive(string &func) {
        int gpu_id = -1;
        for (int i = 0; i < gpu_count_; i++) {
            if (executor_status_[i].status_ == ExecutorStatus_Avail) {
                gpu_id = i;
                break;
            }
        }
        return gpu_id;
    }

    /* round_robin placement */
    int placement(string &func) {
        int gpu_id = -1;
        if (executor_status_[cur_id_].status_ == ExecutorStatus_Avail) {
            gpu_id = cur_id_;
            cur_id_ = (cur_id_ + 1) % gpu_count_;
        }

        if (gpu_id == -1) {
            gpu_id = naive(func);
        }
        return gpu_id;
    }

    ScheduleDecision random_schedule(string &func) {
        vector<int> avail_gpus;
        for (int i = 0; i < gpu_count_; i++) {
            if (executor_status_[i].status_ == ExecutorStatus_Avail) {
                avail_gpus.push_back(i);
            }
        }

        if (avail_gpus.size() <= 0) {
            return {-1, -1, -1};
        }

        if (model_repo_.model_active_device_map_.find(func) != model_repo_.model_active_device_map_.end() && model_repo_.model_active_device_map_[func].size() > 0) {
            for (auto &gpu_id : model_repo_.model_active_device_map_[func]) {
                if (executor_status_[gpu_id].status_ == ExecutorStatus_Avail) { // no swap
                    return {gpu_id, -1, ExecutorStatus_RunLocal};
                }
            }
        }

        // random selection
        int gpu_id = avail_gpus[rand() % avail_gpus.size()];
        return {gpu_id, -1, model_level_to_exec_status(model_level_map_[func])};
    }

    ScheduleDecision interference_aware_schedule(string &func) {
        vector<int> avail_gpus;
        for (int i = 0; i < gpu_count_; i++) {
            if (executor_status_[i].status_ == ExecutorStatus_Avail) {
                avail_gpus.push_back(i);
            }
        }
        std::cout << "Controller -- avail gpus " << avail_gpus.size() << " for func " << func << std::endl;

        if (avail_gpus.size() <= 0) {
            return {-1, -1, -1};
        }

        if (model_repo_.model_active_device_map_.find(func) != model_repo_.model_active_device_map_.end() && model_repo_.model_active_device_map_[func].size() > 0) {
            // std::cout << "Controller -- before g2g check" << std::endl;

            for (auto &gpu_id : model_repo_.model_active_device_map_[func]) {
                if (executor_status_[gpu_id].status_ == ExecutorStatus_Avail) { // no swap
                    // std::cout << "Controller -- non-swap at gpu " << gpu_id << " status " << model_repo_.device_model_info_map_[gpu_id][func].status_ << std::endl;
                    return {gpu_id, -1, ExecutorStatus_RunLocal};
                }
            }
            vector<int> src_gpus;
            std::copy(model_repo_.model_active_device_map_[func].begin(), model_repo_.model_active_device_map_[func].end(), std::back_inserter(src_gpus));

            auto dst_gpu = avail_gpus[0], src_gpu = src_gpus[0];
            for (auto gpu_i: avail_gpus) {
                for (auto gpu_j: src_gpus) {
                    if (nvlink_topology_[gpu_i][gpu_j] == 2) {
                        dst_gpu = gpu_i;
                        src_gpu = gpu_j;
                        break;
                    }
                }
            }
            // auto find_src_dst_gpu = [&] (int &dst_gpu, int &src_gpu, bool is_heavy_model) {
            //     for (auto gpu_i: avail_gpus) {
            //         for (auto gpu_j: src_gpus) {
            //             if (is_heavy_model && nvlink_topology_[gpu_i][gpu_j] == 2) { // fast nvlink for heavy model
            //                 dst_gpu = gpu_i;
            //                 src_gpu = gpu_j;
            //                 return;
            //             }
            //             else if (!is_heavy_model && nvlink_topology_[gpu_i][gpu_j] == 1) { // slow nvlink for light or median model
            //                 dst_gpu = gpu_i;
            //                 src_gpu = gpu_j;
            //                 return;
            //             }
            //         }
            //     } 
            // };

            // find_src_dst_gpu(dst_gpu, src_gpu, model_level_map_[func] == ModelLevel_Heavy);
            
            std::cout << "Controller -- transfer " << func << " from gpu " << src_gpu << " to " << dst_gpu << " avail src size " << src_gpus.size() << std::endl;
            return {dst_gpu, src_gpu, ExecutorStatus_RunNvlink}; // swap from gpu
        }
        else {
            int best_gpu_id = avail_gpus[0], best_status = executor_status_[pcie_pairs_[avail_gpus[0]]].status_;
            for (auto &gpu_id : avail_gpus) {
                if (executor_status_[pcie_pairs_[gpu_id]].status_ < best_status) {
                    best_status = executor_status_[pcie_pairs_[gpu_id]].status_;
                    best_gpu_id = gpu_id;
                }
            }
            // std::cout << "Controller -- select host gpu final " << best_gpu_id << std::endl;
            return {best_gpu_id, -1, model_level_to_exec_status(model_level_map_[func])};
        }
    }


    inline void check_eviction(int gpu_idx, string& func) {
        auto model_to_evict = evictor_->check_and_add(gpu_idx, func, model_repo_.model_host_info_map_[func].total_size_, model_level_map_);
        if (model_to_evict.size() > 0) {
            for (auto &model : model_to_evict) {
                model_repo_.remove_model_info(model, gpu_idx);
                send_to_executor(model, gpu_idx, ExecutorSignal_Unload);
            }
        }
    }

    inline void send_exec_req(string &func, int gpu_id, int status = ExecutorStatus_RunColdStart) {
        if (status == ExecutorStatus_RunColdStart) {
            for (int i = 0; i < gpu_count_; i++) {
                if (i == gpu_id) {
                    executor_status_[i] = {ExecutorStatus_RunColdStart, func};
                    send_to_executor(func, i, ExecutorSignal_Startup);
                }
                else {
                    send_to_executor(func, i, ExecutorSignal_Notify);
                }
            }
        }
        else {
            executor_status_[gpu_id] = {status, func};
            send_to_executor(func, gpu_id, ExecutorSignal_Execute);
        }
    }

    inline void send_load_req(string &func, int dst_gpu, int src_gpu = -1) {
        if (src_gpu >= 0) {
            lock_src_func(func, dst_gpu, src_gpu);
        }
        check_eviction(dst_gpu, func);
        int signal = ExecutorSignal_Load + 2 + src_gpu;
        send_to_executor(func, dst_gpu, signal);
    }

    inline void send_to_executor(string &func, int gpu_id, int signal) {
        executor_vec_[gpu_id]->sig_exec_from_controller(signal, func);
    }

    inline void send_signal_ack(AckType type, int gpu_id = -1) {
        SignalAck ack;
        ack.set_ack(type);
        ack.set_resp(gpu_id);
        
        string ack_serialized;
        ack.SerializeToString(&ack_serialized);
        kZmqUtil->send_string(ack_serialized, &sig_socket_);
    }

    inline void send_signal_ack(zmq::message_t &address,AckType type, int gpu_id = -1) {
        SignalAck ack;
        ack.set_ack(type);
        ack.set_resp(gpu_id);
        
        string ack_serialized;
        ack.SerializeToString(&ack_serialized);
        //kZmqUtil->send_string(ack_serialized, &sig_socket_);
        kZmqUtil->send_string_with_peer(address,ack_serialized, &sig_socket_);
    }

    inline void lock_src_func(string &func, int dst_gpu, int src_gpu) {
        executor_gpu_demand_.insert({dst_gpu, {src_gpu, func}});
        evictor_->lock(src_gpu, func);
    }

    inline void unlock_src_func(int dst_gpu) {
        if (executor_gpu_demand_.find(dst_gpu) != executor_gpu_demand_.end()) {
            evictor_->unlock(executor_gpu_demand_[dst_gpu].first, executor_gpu_demand_[dst_gpu].second);
            executor_gpu_demand_.erase(dst_gpu);
        }
    }

    inline void init_model_readiness_to_remote(string& function,vector<void*>& v_ptrs){
        for(auto& executor : executor_vec_){
            executor->init_model_readiness(function,v_ptrs,manager::ParamReadiness_Remote);
        }
    }

    inline void update_model_param_readiness_to_host(string& function,void* v_ptr){
        for(auto& executor : executor_vec_){
            executor->update_model_param_readiness(function,v_ptr,manager::ParamReadiness_Host);
        }
    }

    inline void listen_to_socket(string& addr,zmq::socket_t* socket){
        if(address_to_socket_map_.find(addr) == address_to_socket_map_.end()){
            address_to_socket_map_.insert(make_pair(addr,socket));
            pollitems_.push_back({static_cast<void*>(socket), 0, ZMQ_POLLIN, 0});
        }
    }

    inline void sync_with_net(){
        for(auto& executor : executor_vec_){
            executor->sync_with_net();
        }
    }
private:
    zmq::socket_t sig_socket_;
    zmq::socket_t socket_;

    vector<zmq::pollitem_t> pollitems_;
    map<string,zmq::socket_t*> address_to_socket_map_;

    logger log_;
    int gpu_count_;
    int cur_id_ = 0;

    map<int, pair<int, string>> executor_gpu_demand_;

    vector<std::shared_ptr<CUDAExecutor>> executor_vec_;
    vector<std::thread> thread_vec_;

    vector<ExecutorStatus> executor_status_;
    queue<pair<string, bool>> req_queue_;

    map<string, int> model_level_map_;

    vector<int> pcie_pairs_;
    vector<vector<int>> nvlink_topology_;

    std::function<ScheduleDecision(string&)> schedule_func_;

    std::shared_ptr<Evictor> evictor_;
    manager::ModelRepo& model_repo_ = manager::ModelRepo::getInstance();
    PeerTCPImpl& h2h_tcp_ = PeerTCPImpl::getInstance();
};

}
#endif  // INCLUDE_CONTROLLER_HPP_