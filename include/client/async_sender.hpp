#ifndef INCLUDE_REQ_SENDER_HPP_
#define  INCLUDE_REQ_SENDER_HPP_

#include "socket_helper.hpp"
#include "safe_ptr.hpp"
#include "utils.hpp"

namespace client{
namespace sender{

zmq::context_t context(1);

const size_t queryBufferSize = 30;
const size_t queryTimewindowInMicroseconds = 50;

using QueryQueue = queue<WapperQuery>;

class AsyncSender {

public:
    AsyncSender(): 
            pusher_(Pusher(&context)),
            pull_socket_(zmq::socket_t(context, ZMQ_PULL)){
            
        auto client_id_str = get_env_var("CLIENT_ID");
        if (client_id_str.size() > 0) {
            client_id_ = stoi(client_id_str);
        } else {
            client_id_ = 0;
        }
        pull_socket_.bind(get_client_addr(client_id_));
        std::cout << "Client address is " << get_client_addr(client_id_) << std::endl;

        // int cur_server_id = get_cur_server_id();
        // pusher_.create(get_server_addr(cur_server_id));

        // push_socket_.connect(get_server_addr(0)); // default server
    }

    ~AsyncSender(){}

    void set_logger(logger log) {
        log_ = log;
    }

    int get_client_id() {
        return client_id_;
    }

    inline void send_async(WapperQuery& req) {
        req.set_need_sync(false);
        req.set_client_id(client_id_);

        query_queue_.push(req);
        // check sending condition
        if (query_queue_.size() >= queryBufferSize || 
                ((!query_queue_.empty()) && std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - last_send_time_).count() >= queryTimewindowInMicroseconds)) {
            send_internal();
            last_send_time_ = std::chrono::system_clock::now();
        }
    }

    template <typename RespType>
    inline RespType send_and_recv(WapperQuery& req) {
        req.set_need_sync(true);
        req.set_client_id(client_id_);
        query_queue_.push(req);
        
        send_internal();
        auto wait_start = std::chrono::system_clock::now();
        string resp_string = kZmqUtil->recv_string(&pull_socket_);
        auto wait_end = std::chrono::system_clock::now();
        log_->info("recv sync response, wait time: {}", std::chrono::duration_cast<std::chrono::microseconds>(wait_end - wait_start).count());

        RespType resp;
        resp.ParseFromString(resp_string);
        return resp;
    }

private:

    inline void send_internal() {
        if (query_queue_.empty()) {
            return;
        }
        QueryList query_list;
        int size = 0;
        while (!query_queue_.empty()) {
            query_list.add_queries()->CopyFrom(query_queue_.front());
            query_queue_.pop();
            size++;
        }
        string req_serialized;
        query_list.SerializeToString(&req_serialized); 
        auto cur_server = get_cur_server_id();
        kZmqUtil->send_string(req_serialized, &pusher_[get_server_addr(cur_server)]);
        // log_->info("Send query list size {} to server {}", size, cur_server);
    }

    inline int get_cur_server_id(){
        auto server_id_str = get_env_var("CUR_SERVER_ID");
        if (server_id_str.size() > 0) {
            return stoi(server_id_str);
        }
        return 0;
    }

private:
    logger log_;

    int client_id_;
    // zmq::socket_t push_socket_;
    Pusher pusher_;
    zmq::socket_t pull_socket_;

    std::thread send_thread_;

    QueryQueue query_queue_;

    std::chrono::time_point<std::chrono::system_clock> last_send_time_;
};


}
}

#endif  // INCLUDE_REQ_SENDER_HPP_

