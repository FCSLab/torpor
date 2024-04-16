#ifndef INCLUDE_BLC_MANAGER_HPP_
#define  INCLUDE_BLC_MANAGER_HPP_

#include "cuda_common.hpp"
#include "utils.hpp"
#include "safe_ptr.hpp"
#include <cstdlib>
#include "model_repo.hpp"
#include <cmath>

namespace server {
namespace manager {


struct DataCpyInfo {
    string function_;
    void* virtual_ptr_;
    void* dst_;
    void* src_;
    size_t size_;
};

const size_t default_block_size = 128 * 1024 * 1024;

const int BlockType_Regular = 0;
const int BlockType_Irregular = 1;

const size_t RegularBlockSize_2MB = 2 * 1024 * 1024;
const size_t RegularBlockSize_20MB = 20 * 1024 * 1024;
const size_t IrregularBlockMinSize = 16 * 1024 * 1024;

struct AddrComparator {
    bool operator()(void* a, void* b) const {
        return (uint64_t) a < (uint64_t) b;
    }
};

struct Block {
    void* base_ptr_;
    int type_;
    size_t avail_size_ = default_block_size;

    std::map<void*, bool, AddrComparator> addr_status_map_;

    Block(void* ptr, int type): base_ptr_(ptr), type_(type) {
        if (type_ == BlockType_Regular) {
            for (int i = 0; i < default_block_size / RegularBlockSize_2MB; ++i) {
                addr_status_map_[(void*) ((uint64_t)base_ptr_ + i * RegularBlockSize_2MB)] = true;
            }
        } 
        else {
            for (int i = 0; i < default_block_size / IrregularBlockMinSize; ++i) {
                addr_status_map_[(void*) ((uint64_t)base_ptr_ + i * IrregularBlockMinSize)] = true;
            }
        }
    }

    void* try_acquire_block(size_t size) {
        void* ptr = nullptr;

        // if (type_ == BlockType_Regular && size == RegularBlockSize_2MB) {
        //     // allocate from end to start
        //     auto itr = addr_status_map_.rbegin();
        //     while (itr != addr_status_map_.rend()) {
        //         if (itr->second == true) {
        //             ptr = itr->first;
        //             itr->second = false;
        //             avail_size_ -= RegularBlockSize_2MB;
        //             break;
        //         }
        //         ++itr;
        //     }
        // }
        // else {
            size_t round_size = type_ == BlockType_Regular ? size : std::pow(2, std::ceil(std::log2(size))); // 20MB or irregular
            size_t step_len = type_ == BlockType_Regular ? size / RegularBlockSize_2MB : round_size / IrregularBlockMinSize;

            auto itr = addr_status_map_.begin();
            while (itr != addr_status_map_.end()) {
                bool is_free = true;
                for (int i = 0; i < step_len; ++i) {
                    if (itr == addr_status_map_.end()) {
                        is_free = false;
                        break;
                    }
                    // std::cout << "In block " << (uint64_t) base_ptr_ << " step " << step_len  << " i " << i << " cur " << (uint64_t) itr->first << " " << itr->second << std::endl;
                    if (itr->second == false) {
                        is_free = false;
                    }
                    itr++;
                }
                if (is_free) {
                    itr = std::prev(itr, step_len);
                    ptr = itr->first;
                    for (int i = 0; i < step_len; ++i) {
                        itr->second = false;
                        itr++;
                    }
                    avail_size_ -= round_size;
                    break;
                }
            }
        // }
        return ptr;
    }

    void release_block(void* ptr, size_t size) {
        size_t round_size = type_ == BlockType_Regular ? size : std::pow(2, std::ceil(std::log2(size)));
        size_t step_len = type_ == BlockType_Regular ? round_size / RegularBlockSize_2MB : round_size / IrregularBlockMinSize;

        auto itr = addr_status_map_.find(ptr);
        if (itr == addr_status_map_.end()) {
            std::cout << "Error: release block " << (uint64_t) ptr << " not found in physical block " << (uint64_t) base_ptr_ << std::endl;
            return;
        }
        for (int i = 0; i < step_len; ++i) {
            itr->second = true;
            itr++;
        }
        avail_size_ += round_size;
    }

    bool operator<(const Block& b) const { return avail_size_ < b.avail_size_; }
    bool operator==(const Block& b) const { return base_ptr_ == b.base_ptr_; }
};

template <typename T>
struct SharedPtrComparator {
    bool operator()(const std::shared_ptr<T>& lhs, const std::shared_ptr<T>& rhs) const {
        return (*lhs) < (*rhs);
    }
};

template <class T, class Container = vector<std::shared_ptr<T>>, class Compare = SharedPtrComparator<T>>
class sp_priority_queue {
protected:
    Container c;
    Compare comp;
public:
    sp_priority_queue(const Container& c_  = Container(),
                            const Compare& comp_ = Compare()) : c(c_), comp(comp_) {
        std::make_heap(c.begin(), c.end(), comp);
    }
    bool empty()       const { return c.empty(); }
    size_t size() const { return c.size(); }
    const std::shared_ptr<T>& top()     const { return c.front(); }
    void push(const std::shared_ptr<T>& x) {
        c.push_back(x);
        std::push_heap(c.begin(), c.end(), comp);
    }
    void pop() {
        std::pop_heap(c.begin(), c.end(), comp);
        c.pop_back();
    }
    void remove(const std::shared_ptr<T>& x) {
        auto it = std::find_if(c.begin(), c.end(), [&x=x](const std::shared_ptr<T>& y) { return *x == *y; });
        if (it != c.end()) {
            c.erase(it);
            std::make_heap(c.begin(), c.end(), comp);
        }
    }
};

using BlockVec = vector<std::shared_ptr<Block>>;
using BlockHeap = sp_priority_queue<Block>;


class BlockManager {
public:
    BlockManager(size_t budget) : alloc_budget_(budget) {}

    virtual void init_allocation(){}
    virtual void acquire_block(void** ptr, size_t size, bool external_call=false) = 0;
    virtual void evict_block(void* ptr, size_t size, bool free_flag=false) = 0;

    virtual void finish_block_alloc() {}
    virtual void finish_release_block() {}

    void set_server_id(int server_id) {
        server_id_ = server_id;
    }

protected:
    size_t alloc_budget_;
    int server_id_;
};

class FixedSizeBlockManager : public BlockManager {
public:
    FixedSizeBlockManager(size_t budget) : BlockManager(budget) {}

    void acquire_block(void** ptr, size_t size, bool external_call = false) {
        if (idle_blocks_[size].size() > 0) {
            // *ptr = idle_blocks_[size][0];
            // idle_blocks_[size].erase(idle_blocks_[size].begin());
            *ptr = idle_blocks_[size].back();
            idle_blocks_[size].pop_back();
        }
        else {
            if (alloc_budget_ >= size) {
                cudaCheck(cudaMalloc(ptr, size));
                alloc_budget_ -= size;
            }
            else {
                // free large idle blocks util we have enough memory
                for (auto &block : idle_blocks_) {
                    // std::cout << "Server " << server_id_ << " -- erase block size " << block.first << std::endl;
                    while (block.second.size() > 0) {
                        void* block_ptr = block.second.back();
                        block.second.pop_back();
                        cudaCheck(cudaFree(block_ptr));
                        alloc_budget_ += block.first;

                        if (alloc_budget_ >= size) {
                            break;
                        }
                    }
                    if (alloc_budget_ >= size) {
                        break;
                    }
                }
                
                if (alloc_budget_ >= size) {
                    // std::cout << "Server " << server_id_ << " -- alloc block size " << size << " budget " << alloc_budget_ << std::endl;
                    cudaCheck(cudaMalloc(ptr, size));
                    alloc_budget_ -= size;
                }
                else {
                    if (external_call) {
                        cudaCheck(cudaMalloc(ptr, size));
                    }
                    else {
                        // still not enough memory, return nullptr
                        // *ptr = nullptr;
                        cudaCheck(cudaMalloc(ptr, size));
                        // log_->warn("No enough memory when acquiring block of size {} budget {}", size, alloc_budget_);
                        std::cout << "Server " << server_id_ << " -- Warning: no enough memory when acquiring block of size " << size << " budget " << alloc_budget_ << std::endl;
                    }
                }
            }
        }
    }

    void evict_block(void* ptr, size_t size, bool free_flag=false) {
        if (free_flag) {
            cudaCheck(cudaFree(ptr));
            alloc_budget_ += size;
        }
        else {
            idle_blocks_[size].push_back(ptr);
            // this will take considerable overhead !!!
            // cudaCheck(cudaMemsetAsync(ptr, 0, size, trans_stream_));
        }
    }

private:
    ordered_map<size_t, vector<void*>> idle_blocks_;
};

class BuddyBasedBlockManager : public BlockManager {
public:
    BuddyBasedBlockManager(size_t budget) : BlockManager(budget) {}

    void init_allocation() {
        // allocate budget using default block size
        size_t num_blocks = alloc_budget_ / default_block_size;
        for (size_t i = 0; i < num_blocks; i++) {
            void* ptr;
            cudaCheck(cudaMalloc(&ptr, default_block_size));
            free_block_pool_.push_back(ptr);
        }
        std::cout << "Server " << server_id_  << " -- Allocated " << num_blocks << " blocks of size " << default_block_size << std::endl;
    }

    void finish_block_alloc() {
        // reset active_blocks
        for (auto &block : active_regular_blocks_) {
            regular_block_priority_heap_.push(std::move(block));
        }
        active_regular_blocks_.clear();

        for (auto &block : active_irregular_blocks_) {
            irregular_block_priority_heap_.push(std::move(block));
        }
        active_irregular_blocks_.clear();
    }

    void acquire_block(void** ptr, size_t size, bool external_call = false) {
        if (size == RegularBlockSize_2MB || size == RegularBlockSize_20MB) {
            acquire_block_temp(ptr, size, active_regular_blocks_, regular_block_priority_heap_, true);
        }
        else if (size <= default_block_size) {
            acquire_block_temp(ptr, size, active_irregular_blocks_, irregular_block_priority_heap_, false);
        }
        else {
            // expect this never happens
            std::cout << "Server " << server_id_ << " -- acquire_block size too large " << size << std::endl;
            // log_->warn("acquire_block size {}", size);
            cudaCheck(cudaMalloc(ptr, size));
        }
    }

    inline void acquire_block_temp(void** ptr, size_t size, BlockVec& active_blocks_, BlockHeap& block_heap_, bool is_regular) {
        void* ptr_to_return = nullptr;

        for (auto &block : active_blocks_) {
            ptr_to_return = block->try_acquire_block(size);
            if (ptr_to_return != nullptr) {
                *ptr = ptr_to_return;
                requested_to_block_map_[ptr_to_return] = block;
                // std::cout << "Server " << server_id_ << " -- acquire " << size << " block " << (uint64_t) ptr_to_return << " from active block " << (uint64_t) block->base_ptr_ << " len " << active_blocks_.size() << std::endl;
                return;
            }
        }

        // if no block is available, put a new block into active_blocks_
        while (block_heap_.size() > 0 && block_heap_.top()->avail_size_ >= size) {
            active_blocks_.push_back(std::move(block_heap_.top()));
            block_heap_.pop();
            // std::cout << "Server " << server_id_ << " -- check heap block " << (uint64_t) active_blocks_.back()->base_ptr_ << " at top " << active_blocks_.back()->avail_size_ << " len " << block_heap_.size() << std::endl;


            ptr_to_return = active_blocks_.back()->try_acquire_block(size);
            if (ptr_to_return != nullptr) {
                *ptr = ptr_to_return;
                requested_to_block_map_[ptr_to_return] = active_blocks_.back();
                return;
            }
        }

        // fetch a new block from free block pool
        if (free_block_pool_.size() > 0) {
            active_blocks_.push_back(std::make_shared<Block>(free_block_pool_.back(), is_regular? BlockType_Regular : BlockType_Irregular));
            free_block_pool_.pop_back();

            ptr_to_return = active_blocks_.back()->try_acquire_block(size);
            if (ptr_to_return != nullptr) {
                *ptr = ptr_to_return;
                requested_to_block_map_[ptr_to_return] = active_blocks_.back();
                // std::cout << "Server " << server_id_ << " -- acquire " << size << " block " << (uint64_t) ptr_to_return << " from free block " << (uint64_t) active_blocks_.back()->base_ptr_ << " len " << free_block_pool_.size() << std::endl;
                return;
            }
            else {
                // expect this never happens
                std::cout << "Server " << server_id_ << " -- Error: cannot acquire block " << size << " from a free physical block" << std::endl;
                exit(1);
            }
        }
        else {
            std::cout << "Server " << server_id_ << " -- Error: no free physical block" << std::endl;
            exit(1);
        }
    }

    void finish_release_block() {
        // resort block heaps and collect free block to pool
        for (auto &block : deactive_regular_blocks_) {
            if (block->avail_size_ == default_block_size) {
                free_block_pool_.push_back(block->base_ptr_);
                // std::cout << "Server " << server_id_ << " -- free regular block " << (uint64_t) block->base_ptr_ << std::endl;
            }
            else {
                // std::cout << "Server " << server_id_ << " -- push regular block " << (uint64_t) block->base_ptr_ << " avail size " << block->avail_size_ << std::endl;
                regular_block_priority_heap_.push(std::move(block));
            }
        }
        deactive_regular_blocks_.clear();

        for (auto &block : deactive_irregular_blocks_) {
            if (block->avail_size_ == default_block_size) {
                free_block_pool_.push_back(block->base_ptr_);
                // std::cout << "Server " << server_id_ << " -- free irregular block " << (uint64_t) block->base_ptr_ << std::endl;
            }
            else {
                // std::cout << "Server " << server_id_ << " -- push irregular block " << (uint64_t) block->base_ptr_ << " avail size " << block->avail_size_ << std::endl;
                irregular_block_priority_heap_.push(std::move(block));
            }
        }
        deactive_irregular_blocks_.clear();
    }

    void evict_block(void* ptr, size_t size, bool free_flag=false) {
        if (requested_to_block_map_.find(ptr) != requested_to_block_map_.end()) {
            requested_to_block_map_[ptr]->release_block(ptr, size);
            // std::cout << "Server " << server_id_ << " -- evict " << size << " block " << (uint64_t) ptr << " from " << (uint64_t) requested_to_block_map_[ptr]->base_ptr_ << std::endl;
            if (size == RegularBlockSize_2MB || size == RegularBlockSize_20MB) {
                if (std::find_if(deactive_regular_blocks_.begin(), deactive_regular_blocks_.end(), 
                        [&requested_to_block_map_=requested_to_block_map_, ptr] (std::shared_ptr<Block> const& i) {return i.get() == requested_to_block_map_[ptr].get();} ) == deactive_regular_blocks_.end()) {
                    deactive_regular_blocks_.push_back(requested_to_block_map_[ptr]);
                    regular_block_priority_heap_.remove(requested_to_block_map_[ptr]);
                }
            }
            else if (size <= default_block_size) {
                if (std::find_if(deactive_irregular_blocks_.begin(), deactive_irregular_blocks_.end(), 
                        [&requested_to_block_map_=requested_to_block_map_, ptr] (std::shared_ptr<Block> const& i) {return i.get() == requested_to_block_map_[ptr].get();} ) == deactive_irregular_blocks_.end()) {
                    deactive_irregular_blocks_.push_back(requested_to_block_map_[ptr]);
                    irregular_block_priority_heap_.remove(requested_to_block_map_[ptr]);
                }
            }
            requested_to_block_map_.erase(ptr);
        }
        else {
            std::cout << "Server " << server_id_ << " -- Error: cannot find physical block for ptr " << ptr << std::endl;
        }
    }

private:
    vector<void*> free_block_pool_;

    BlockVec active_regular_blocks_;
    BlockVec deactive_regular_blocks_;
    BlockHeap regular_block_priority_heap_;
    BlockVec active_irregular_blocks_;
    BlockVec deactive_irregular_blocks_;
    BlockHeap irregular_block_priority_heap_;

    map<void*, std::shared_ptr<Block>> requested_to_block_map_;
};


}
}
#endif  // INCLUDE_MEM_MANAGER_HPP_
