#ifndef INCLUDE_OUR_UTILS_HPP_
#define INCLUDE_OUR_UTILS_HPP_

#include <queue>
#include <mutex>
#include <condition_variable>
#include <iomanip>
#include <unistd.h>

#include "types.hpp"
#include <stdlib.h>

inline string get_env_var(const string& name) {
    const char * val = std::getenv(name.c_str());
    if ( val == nullptr ) {
        return "";
    }
    return string(val);
}


// From: https://stackoverflow.com/questions/15278343/c11-thread-safe-queue
template <class T>
class BlockingQueue{
public:
  BlockingQueue()
    : q()
    , m()
    , c()
  {}

  ~BlockingQueue(){}

  // Add an element to the queue.
  void enqueue(T t) {
    std::lock_guard<std::mutex> lock(m);
    q.push(t);
    c.notify_one();
  }

  // Get the "front"-element.
  // If the queue is empty, wait till a element is avaiable.
  T dequeue() {
    std::unique_lock<std::mutex> lock(m);
    while(q.empty())
    {
      // release lock as long as the wait and reaquire it afterwards.
      c.wait(lock);
    }
    T val = q.front();
    q.pop();
    return val;
  }

private:
  std::queue<T> q;
  mutable std::mutex m;
  std::condition_variable c;
};

template <class T>
using blocking_queue = std::shared_ptr<BlockingQueue<T>>;

template <class T>
inline blocking_queue<T> create_blocking_queue() {
    return std::make_shared<BlockingQueue<T>>();
}


const int ExecutorSignal_Notify = 0;
const int ExecutorSignal_Startup = 1;
const int ExecutorSignal_Execute = 2;
const int ExecutorSignal_Unload = 3;
const int ExecutorSignal_Load = 4;


#endif  // INCLUDE_OUR_UTILS_HPP_
