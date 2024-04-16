#ifndef INCLUDE_SOCKET_HELPER_HPP_
#define  INCLUDE_SOCKET_HELPER_HPP_

#include "zmq/zmq_util.hpp"
#include <map>

ZmqUtil zmq_util;
ZmqUtilInterface *kZmqUtil = &zmq_util;

inline string get_client_addr(int client_id) {
  return "ipc:///cuda/client_" + std::to_string(client_id);
}

inline string get_server_addr(int server_id) {
  return "ipc:///cuda/server_" + std::to_string(server_id);
}

inline string get_signal_addr(int server_id) {
    return "ipc:///cuda/signal_" + std::to_string(server_id);
}

inline string get_model_send_addr(int server_id) {
    return "ipc:///cuda/model_send_" + std::to_string(server_id);
}

inline string get_model_recv_addr(int server_id) {
    return "ipc:///cuda/model_recv_" + std::to_string(server_id);
}

inline string get_scheduler_addr() {
    return "ipc:///cuda/scheduler";
}

class Pusher {
public:
    explicit Pusher(zmq::context_t* zmq_context) : context_(zmq_context) {}

    zmq::socket_t& At(const string& addr) {
        if (cache_.find(addr) == cache_.end()) {
            create(addr);
        }
        auto iter = cache_.find(addr);
        return iter->second;
    }

    zmq::socket_t& operator[](const string& addr) {
        return At(addr);
    }

    void create(const string& addr) {
        if (cache_.find(addr) == cache_.end()) {
            zmq::socket_t socket(*context_, ZMQ_PUSH);
            socket.connect(addr);
            cache_.insert(std::make_pair(addr, std::move(socket)));
        }
    }

    void clear() {
        cache_.clear();
    }

private:
    zmq::context_t* context_;
    std::map<const string, zmq::socket_t> cache_;
};

#endif  // INCLUDE_SOCKET_HELPER_HPP_

