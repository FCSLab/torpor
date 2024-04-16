#include <stdio.h>
#include <iostream>
#include <sstream>
#include <chrono>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    int total_sz = 128 * 1000 * 1000;
    int count = 1;

    if (argc > 1) {
        count = atoi(argv[1]);
    }

    size_t sz = total_sz / count;

    char* a = (char*) malloc(total_sz);
    char* b = (char*) malloc(total_sz);

    memset(a, 0, total_sz);
    for (int i = 0; i < count; i++){
        memcpy(b + i * sz, a + i * sz, sz);
    }

    std::vector<int> elasped_vec;
    for (int time = 0; time < 10; time++){
        memset(a, time, total_sz);
        auto start_t = std::chrono::system_clock::now();
        for (int i = 0; i < count; i++){
            memcpy(b + i * sz, a + i * sz, sz);
        }
        auto end_t = std::chrono::system_clock::now();
        auto mem_cpy_elasped = std::chrono::duration_cast<std::chrono::microseconds> (end_t - start_t).count();
        elasped_vec.push_back(mem_cpy_elasped);
    }
    int sum = 0;
    for (auto e: elasped_vec) sum += e;
    auto avg =  (float)sum / elasped_vec.size();

    auto throughput = total_sz / 1024 / 1024 / avg * 1000 * 1000;

    std::cout << "Memcpy buffer size " << sz << " avg_latency " << std::to_string(avg) << " throughput " << std::to_string(throughput) << std::endl;
    for (auto e: elasped_vec) std::cout << e << " ";
    std::cout << std::endl;

}
