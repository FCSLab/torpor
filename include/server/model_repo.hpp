#ifndef INCLUDE_MODEL_REPO_HPP_
#define  INCLUDE_MODEL_REPO_HPP_

#include "cuda_common.hpp"
#include "utils.hpp"
#include "safe_ptr.hpp"
#include <cstdlib>

namespace server {
namespace manager {

struct GeneralBlockInfo {
    size_t size_;
    bool is_model_;
    // void* cuda_addr_;
    set<void*> ephe_data_ptrs_;
};

struct ModelBlockInfo {
    void* host_addr_;
    // void* pinned_addr_;
    set<void*> virtual_param_ptrs_;
};

struct ModelParamInfo {
    size_t size_;
    void* host_addr_;
    // void* pinned_addr_;
    // void* cuda_addr_;
};

const int ModelStatus_Full = 0;
const int ModelStatus_Incomplete = 1; // during generation at start
const int ModelStatus_Partial = 2; // partial model parameter ready in device
const int ModelStatus_NoData = 3; // model parameter not ready

struct HostModelInfo {
    map<void*, GeneralBlockInfo> virtual_blocks_;
    map<void*, void*> virtual_data_block_mapping_;

    map<void*, ModelBlockInfo> model_block_info_;
    map<void*, ModelParamInfo> model_param_info_;

    int status_ = ModelStatus_Incomplete;
    size_t total_size_ = 0;
};

struct BlockDeviceInfo {
    bool in_cuda_;
    bool in_pin_;
    void* pinned_addr_;
    void* cuda_addr_;
};

struct ParamDeviceInfo {
    void* pinned_addr_;
    void* cuda_addr_;
};

struct DeviceModelInfo {
    map<void*, BlockDeviceInfo> block_virtual_to_cuda_;
    map<void*, ParamDeviceInfo> param_virtual_to_cuda_;

    int status_ = ModelStatus_Incomplete;
};

class ModelRepo {
public:
    static ModelRepo& getInstance() {
        static ModelRepo repo;
        return repo;
    }

    void init_device_info(int device_id) {
        device_model_info_map_[device_id] = {};
    }

    void init_device_model_info(int device_id, string &function) {
        device_model_info_map_[device_id][function] = {{}, {}, ModelStatus_NoData};
    }

    void add_model_info(string &function, int gpu_id) {
        device_model_info_map_[gpu_id][function].status_ = ModelStatus_Full;
        model_active_device_map_[function].insert(gpu_id);
    }

    void remove_model_info(string &function, int gpu_id) {
        device_model_info_map_[gpu_id][function].status_ = ModelStatus_NoData;
        model_active_device_map_[function].erase(gpu_id);
    }

    void start_model_load(string &function) {
        model_host_info_map_[function].status_ = ModelStatus_Incomplete;
    }

    void complete_model_load(string &function) {
        model_host_info_map_[function].status_ = ModelStatus_Full;
    }


    // write on internal data structure
    void check_add_model_access_order(string &function, void* dev_addr) {
        if (model_access_order_map_.find(function) == model_access_order_map_.end()) {
            model_access_order_map_[function] = {};
        }
        if (std::find(model_access_order_map_[function].begin(), model_access_order_map_[function].end(), dev_addr) == model_access_order_map_[function].end()) {
            model_access_order_map_[function].push_back(dev_addr);
        }
    }

    bool find_block_ephe_ptr_mapping(string& function, void* addr) {
        if (model_host_info_map_[function].virtual_data_block_mapping_.find(addr) != model_host_info_map_[function].virtual_data_block_mapping_.end()) return true;

        for (auto &info : model_host_info_map_[function].virtual_blocks_) {
            if ((uint64_t) addr >= (uint64_t) info.first && (uint64_t) addr < (uint64_t) info.first + info.second.size_) {
                model_host_info_map_[function].virtual_data_block_mapping_[addr] = info.first;
                model_host_info_map_[function].virtual_blocks_[info.first].ephe_data_ptrs_.insert(addr);
                return true;
            }
        }
        return false;
    }

    void load_model_param_to_host(string& function,void* v_ptr,const string &payload,size_t size){
        void* host_ptr = malloc_block_in_host(size);
        model_host_info_map_[function].model_block_info_[v_ptr] = {host_ptr, {}};
        model_host_info_map_[function].model_block_info_[v_ptr].virtual_param_ptrs_.insert(v_ptr);
        memcpy(host_ptr, &payload[0], size);
        model_host_info_map_[function].model_param_info_[v_ptr] = {size, host_ptr};
        model_host_info_map_[function].virtual_data_block_mapping_[v_ptr] = v_ptr;
    }

    void malloc_block_for_model(void* ptr, size_t size, bool load_model_flag, string& function){
        if (load_model_flag) {
            void* host_ptr = malloc_block_in_host(size);
            model_host_info_map_[function].model_block_info_[ptr] = {host_ptr, {}};
        }

        model_host_info_map_[function].virtual_blocks_[ptr] = {size, load_model_flag, {}};
        model_host_info_map_[function].total_size_ += size;
    }

    // In general users do not explicitly call cudaFree
    void free_block_for_model(void* ptr, string& function){
        for (auto& ephe_data_ptr : model_host_info_map_[function].virtual_blocks_[ptr].ephe_data_ptrs_) {
            model_host_info_map_[function].virtual_data_block_mapping_.erase(ephe_data_ptr);
        }
        model_host_info_map_[function].virtual_blocks_.erase(ptr);     
    }

    bool load_model_h2d_at_start(void* dst, const string &payload, size_t size, string& function){
        uint64_t host_addr, block_addr = 0;
        for (auto &malloc_info : model_host_info_map_[function].model_block_info_) {
            if ((uint64_t) dst >= (uint64_t) malloc_info.first && (uint64_t) dst < (uint64_t) malloc_info.first + model_host_info_map_[function].virtual_blocks_[malloc_info.first].size_) {
                host_addr = (uint64_t) dst - (uint64_t) malloc_info.first + (uint64_t) malloc_info.second.host_addr_;
                block_addr = (uint64_t) malloc_info.first;
                malloc_info.second.virtual_param_ptrs_.insert(dst);
                break;
            }
        }
        if (host_addr > 0) {
            memcpy((void*)host_addr, &payload[0], size);
            model_host_info_map_[function].model_param_info_[dst] = {size, (void*) host_addr};
            model_host_info_map_[function].virtual_data_block_mapping_[dst] = (void*) block_addr;
            return true;
        }
        return false;
    }

private:
    ModelRepo() {}

    inline void* malloc_block_in_host(size_t size){
        // use pagable memory
        void* host_ptr = malloc(size);
        return host_ptr;
    }

public:
    map<string, vector<void*>> model_access_order_map_;
    map<string, HostModelInfo> model_host_info_map_;
    map<string,bool> model_meta_data_info_map_;

    map<int, map<string, DeviceModelInfo>> device_model_info_map_;
    map<string, set<int>> model_active_device_map_;
};

}
}

#endif  // INCLUDE_MODEL_REPO_HPP_
