#ifndef INCLUDE_EVICTOR_HPP_
#define  INCLUDE_EVICTOR_HPP_

#include <algorithm>
#include "utils.hpp"
#include <list>

namespace server {

const int ModelLevel_Light = 0;
const int ModelLevel_Median = 1;
const int ModelLevel_Heavy = 2;

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

}

#endif  // INCLUDE_EVICTOR_HPP_