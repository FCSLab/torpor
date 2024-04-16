#include "controller.hpp"

int main(int argc, char** argv) {
    string log_name = "log_cuda_server";
    string log_file = log_name + ".txt";
    auto log = spdlog::basic_logger_mt(log_name, log_file, true);
    log->set_pattern("[%Y-%m-%d %T.%f] [%l] %v");
    
    log->flush_on(spdlog::level::info);

    zmq::context_t context(1);
    int maxSocket = 60000;
    auto res = context.setctxopt(ZMQ_MAX_SOCKETS, maxSocket);
    if (res != 0) {
        log->error("Set context max socket to {} failed", maxSocket);
    }
    else {
        log->info("Set context max socket to {} success", maxSocket);
    }
    std::cout << "Set context max socket to " << maxSocket << " res " << res << std::endl;

    server::Controller server_controller(log, &context);
    std::cout << "Server start" << std::endl;

    server_controller.start();
    return 0;
}



