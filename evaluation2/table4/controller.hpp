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

namespace server {

const int ExecutorStatus_Avail = -1;
const int ExecutorStatus_RunLocal = 0;
const int ExecutorStatus_RunNvlink = 1;
const int ExecutorStatus_RunPcleLight = 2;
const int ExecutorStatus_RunPcleMedian = 3;
const int ExecutorStatus_RunPcleHeavy = 4;
const int ExecutorStatus_RunColdStart = 5;

const int ModelLevel_Light = 0;
const int ModelLevel_Median = 1;
const int ModelLevel_Heavy = 2;

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

class Evictor {
public:
    Evictor(int gpu_count, size_t budget): gpu_count_(gpu_count), budget_(budget) {}

    virtual vector<string> check_and_add(int gpu_idx, string& func, size_t sz, map<string, int>& model_level_map) = 0;
    virtual void force_evict(int gpu_idx, string& func) = 0;
    virtual void lock(int gpu_idx, string& func) = 0;
    virtual void unlock(int gpu_idx, string& func) = 0;

protected:
    int gpu_count_;
    size_t budget_;
};

class InterferenceAwareEvictor : public Evictor {
public:
    InterferenceAwareEvictor(int gpu_count, size_t budget): Evictor(gpu_count, budget) {
        for (int i = 0; i < gpu_count_; i++){
            items_first_class.push_back(std::list<pair<string, size_t>>());
            index_first_class.push_back(map<string, typename std::list<pair<string, size_t>>::iterator>());
            items_second_class.push_back(std::list<pair<string, size_t>>());
            index_second_class.push_back(map<string, typename std::list<pair<string, size_t>>::iterator>());
            locked_items.push_back(set<string>());
            cur_size.push_back(0);
        }
    }

    vector<string> check_and_add(int gpu_idx, string& func, size_t sz, map<string, int>& model_level_map) {
        // std::cout << "Controller -- check_and_add " << func << " on gpu " << gpu_idx << std::endl;
        if (model_to_gpus.find(func) != model_to_gpus.end() && model_to_gpus[func].find(gpu_idx) != model_to_gpus[func].end()) {
            // fresh existing item
            auto itr = index_first_class[gpu_idx].find(func);
            if (itr != index_first_class[gpu_idx].end()) {
                items_first_class[gpu_idx].splice(items_first_class[gpu_idx].begin(), items_first_class[gpu_idx], itr->second);
            }
            else {
                itr = index_second_class[gpu_idx].find(func);
                if (itr != index_second_class[gpu_idx].end()) {
                    items_second_class[gpu_idx].splice(items_second_class[gpu_idx].begin(), items_second_class[gpu_idx], itr->second);
                }
                else {
                    std::cout << "Controller -- warning: evictor data inconsistency for " << func << " on gpu " << gpu_idx << std::endl;
                }
            }
            
            return vector<string>();
        }

        vector<string> model_to_evict;

        // std::cout << "Controller -- check_and_add " << func << " search second class" << std::endl;
        while (cur_size[gpu_idx] + sz >= budget_) {
            // find the first unlocked item from second class
            bool find_item = false;
            for (auto it = items_second_class[gpu_idx].rbegin(); it != items_second_class[gpu_idx].rend(); ++it) {
                if (locked_items[gpu_idx].find(it->first) == locked_items[gpu_idx].end()) {
                    model_to_evict.push_back(it->first);
                    cur_size[gpu_idx] -= it->second;

                    index_second_class[gpu_idx].erase(it->first);
                    items_second_class[gpu_idx].erase(std::next(it).base());
                    find_item = true;
                    break;
                }
            }
            if (!find_item) {
                break;
            }
        }

        // std::cout << "Controller -- check_and_add " << func << " search first class" << std::endl;
        while (cur_size[gpu_idx] + sz >= budget_) {
            // find the first unlocked item from first class
            for (auto it = items_first_class[gpu_idx].rbegin(); it != items_first_class[gpu_idx].rend(); ++it) {
                if (locked_items[gpu_idx].find(it->first) == locked_items[gpu_idx].end()) {
                    model_to_evict.push_back(it->first);
                    cur_size[gpu_idx] -= it->second;

                    index_first_class[gpu_idx].erase(it->first);
                    items_first_class[gpu_idx].erase(std::next(it).base());
                    break;
                }
            }
        }

        if (model_level_map[func] >= ModelLevel_Median && model_to_gpus[func].size() <= 0) {
            // insert to the first class
            items_first_class[gpu_idx].emplace_front(func, sz);
            index_first_class[gpu_idx].emplace(func, items_first_class[gpu_idx].begin());
        }
        else {
            // insert to the second class
            items_second_class[gpu_idx].emplace_front(func, sz);
            index_second_class[gpu_idx].emplace(func, items_second_class[gpu_idx].begin());

            if (model_level_map[func] >= ModelLevel_Median && model_to_gpus[func].size() > 0) {
                // std::cout << "Controller -- check_and_add: class switch to second " << func << " other gpus " <<  model_to_gpus[func].size() << std::endl;

                for (auto &other_g : model_to_gpus[func]) {
                    // switch the first class to the second class on other gpu
                    auto itr = index_first_class[other_g].find(func);
                    if (itr != index_first_class[other_g].end()) {
                        items_second_class[other_g].splice(items_second_class[other_g].begin(), items_first_class[other_g], itr->second);
                        index_second_class[other_g].emplace(func, items_second_class[other_g].begin());
                        index_first_class[other_g].erase(itr);
                    }
                }
            }
        }
        model_to_gpus[func].insert(gpu_idx);
        cur_size[gpu_idx] += sz;

        // TODO!! same model check!!

        // update status of removing models
        for (auto &mod : model_to_evict) {
            if (model_to_gpus[mod].find(gpu_idx) != model_to_gpus[mod].end()) {
                model_to_gpus[mod].erase(gpu_idx);
            }
            if (model_level_map[mod] >= ModelLevel_Median && model_to_gpus[mod].size() == 1) {
                // switch the second class to the first class on other gpu
                auto other_g = *model_to_gpus[mod].begin();
                // std::cout << "Controller -- check_and_add: class switch to first " << mod << " only gpu " <<  other_g << std::endl;

                auto itr = index_second_class[other_g].find(mod);
                if (itr != index_second_class[other_g].end()) {
                    items_first_class[other_g].splice(items_first_class[other_g].begin(), items_second_class[other_g], itr->second);
                    index_first_class[other_g].emplace(mod, items_first_class[other_g].begin());
                    index_second_class[other_g].erase(itr);
                }
            }
        }

        return model_to_evict;
    }

    void force_evict(int gpu_idx, string& func) {
        auto itr = index_first_class[gpu_idx].find(func);
        if (itr != index_first_class[gpu_idx].end()) {
            cur_size[gpu_idx] -= itr->second->second;
            items_first_class[gpu_idx].erase(itr->second);
            index_first_class[gpu_idx].erase(itr);
        }
        else {
            itr = index_second_class[gpu_idx].find(func);
            if (itr != index_second_class[gpu_idx].end()) {
                cur_size[gpu_idx] -= itr->second->second;
                items_second_class[gpu_idx].erase(itr->second);
                index_second_class[gpu_idx].erase(itr);
            }
        }

        if (model_to_gpus[func].find(gpu_idx) != model_to_gpus[func].end()) {
            model_to_gpus[func].erase(gpu_idx);
        }
    }

    void lock(int gpu_idx, string& func) {
        locked_items[gpu_idx].insert(func);
    }

    void unlock(int gpu_idx, string& func) {
        if (locked_items[gpu_idx].find(func) != locked_items[gpu_idx].end()) {
            locked_items[gpu_idx].erase(func);
        }
    }

private:
    vector<set<string>> locked_items; 

    vector<std::list<pair<string, size_t>>> items_first_class;
    vector<map<string, typename std::list<pair<string, size_t>>::iterator>> index_first_class;
    vector<std::list<pair<string, size_t>>> items_second_class;
    vector<map<string, typename std::list<pair<string, size_t>>::iterator>> index_second_class;
    map<string, set<int>> model_to_gpus;
    vector<size_t> cur_size;
};

class LRUEvictor : public Evictor {
public:
    LRUEvictor(int gpu_count, size_t budget): Evictor(gpu_count, budget) {
        for (int i = 0; i < gpu_count_; i++){
            items.push_back(std::list<pair<string, size_t>>());
            index.push_back(map<string, typename std::list<pair<string, size_t>>::iterator>());
            locked_items.push_back(set<string>());
            cur_size.push_back(0);
        }
    }

    vector<string> check_and_add(int gpu_idx, string& func, size_t sz, map<string, int>& model_level_map) {
        // update order with this access
        auto itr = index[gpu_idx].find(func);
        if (itr != index[gpu_idx].end()) {
            items[gpu_idx].splice(items[gpu_idx].begin(), items[gpu_idx], itr->second);
            return vector<string>();
        }

        vector<string> model_to_evict;
        // evict LRU items
        while (cur_size[gpu_idx] + sz >= budget_) {

            // find the first unlocked item
            for (auto it = items[gpu_idx].rbegin(); it != items[gpu_idx].rend(); ++it) {
                if (locked_items[gpu_idx].find(it->first) == locked_items[gpu_idx].end()) {
                    model_to_evict.push_back(it->first);
                    cur_size[gpu_idx] -= it->second;

                    index[gpu_idx].erase(it->first);
                    items[gpu_idx].erase(std::next(it).base());
                    break;
                }
            }
        }

        items[gpu_idx].emplace_front(func, sz);
        index[gpu_idx].emplace(func, items[gpu_idx].begin());
        cur_size[gpu_idx] += sz;

        return model_to_evict;
    }

    void force_evict(int gpu_idx, string& func) {
        auto itr = index[gpu_idx].find(func);
        if (itr != index[gpu_idx].end()) {
            cur_size[gpu_idx] -= itr->second->second;
            items[gpu_idx].erase(itr->second);
            index[gpu_idx].erase(itr);
        }
        else {
            std::cout << "Controller -- evictor force_evict " << func << " not found" << std::endl;
        }
    }

    void lock(int gpu_idx, string& func) {
        auto itr = index[gpu_idx].find(func);
        if (itr != index[gpu_idx].end()) {
            locked_items[gpu_idx].insert(func);
        }
    }

    void unlock(int gpu_idx, string& func) {
        if (locked_items[gpu_idx].find(func) != locked_items[gpu_idx].end()) {
            locked_items[gpu_idx].erase(func);
        }
    }

private:
    vector<set<string>> locked_items; 

    vector<std::list<pair<string, size_t>>> items;
    vector<map<string, typename std::list<pair<string, size_t>>::iterator>> index;
    vector<size_t> cur_size;
};


class Controller {
public:
    Controller(logger log, zmq::context_t *context): 
            log_(log),
            sig_socket_(zmq::socket_t(*context, ZMQ_REP)) {
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

        sig_socket_.bind(get_signal_addr(0));

        pollitems_ = {
            {static_cast<void*>(sig_socket_), 0, ZMQ_POLLIN, 0} ,
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
                auto req_serialized = kZmqUtil->recv_string(&sig_socket_);
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
                        send_signal_ack(AckType::OK, gpu_idx);
                    }
                }
                else if (req.type() == RequestType::Execute) {
                    auto hint = req.payload();
                    if (hint.size() > 0) {
                        // externel test
                        auto gpu_idx = std::stoi(hint);
                        // test p2p
                        // send_load_req(func, gpu_idx, gpu_idx == 0 && model_repo_.model_active_device_map_[func].find(src_gpu_for_test) != model_repo_.model_active_device_map_[func].end()? src_gpu_for_test : -1);
                        // send_load_req(func, gpu_idx);
			if (gpu_idx == 2) { send_load_req(func, gpu_idx, 1); }
			else{ send_load_req(func, gpu_idx); }
                        send_exec_req(func, gpu_idx, model_level_to_exec_status(model_level_map_[func]));
                        send_signal_ack(AckType::OK, gpu_idx);
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
                            send_signal_ack(AckType::OK, decision.gpu_idx_);
                        }
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
                    send_signal_ack(AckType::OK);
                }
                else if (req.type() == RequestType::Load) {
                    // test
                    for (int i = 0; i < gpu_count_; i++) {
                        send_load_req(func, i);
                        model_repo_.add_model_info(func, i);
                    }
                    send_signal_ack(AckType::OK);
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
        int signal = ExecutorSignal_Load + 1 + src_gpu;
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


private:
    zmq::socket_t sig_socket_;
    vector<zmq::pollitem_t> pollitems_;

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
};

}
#endif  // INCLUDE_CONTROLLER_HPP_
