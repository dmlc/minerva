#pragma once
#include <list>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <cstdio>
#include "bool_flag.h"
#include "common.h"

template<typename T> class ConcurrentBlockingQueue {
 public:
  ConcurrentBlockingQueue() : exit_now_(false) {
  }
  void Push(const T& e) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push_back(e);
    if (queue_.size() == 1) {
      cv_.notify_all();
    }
  }
  bool Pop(T& rv) {
    std::unique_lock<std::mutex> lock(mutex_);
    while (queue_.empty() && !exit_now_.Read()) {
      cv_.wait(lock);
    }
    if (!exit_now_.Read()) {
      rv = queue_.front();
      queue_.pop_front();
      return false;
    } else {
      return true;
    }
  }
  std::list<T> PopAll() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::list<T> rv;
    rv.swap(queue_);
    return rv;
  }
  // Call `SignalForKill` before destruction
  void SignalForKill() {
    std::unique_lock<std::mutex> lock(mutex_);
    exit_now_.Write(true);
    cv_.notify_all();
  }
  size_t QueueSize() {
    std::unique_lock<std::mutex> lock(mutex_);
    return queue_.size();
  }

 private:
  std::list<T> queue_;
  std::mutex mutex_;
  std::condition_variable cv_;
  BoolFlag exit_now_;
  DISALLOW_COPY_AND_ASSIGN(ConcurrentBlockingQueue);
};

