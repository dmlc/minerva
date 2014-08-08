#pragma once
#include "common/common.h"
#include "common/concurrent_blocking_queue.h"
#include <vector>
#include <functional>
#include <thread>
#include <utility>

namespace minerva {

class ThreadPool {
 public:
  typedef std::function<void()> Task;
  ThreadPool(size_t numthreads) {
    for(size_t thrid = 0; thrid < numthreads; ++thrid) {
      workers_.push_back(std::thread(&ThreadPool::SimpleWorker, this, thrid));
    }
  }
  ~ThreadPool() {
    task_queue_.SignalForKill();
    for(auto & w : workers_) {
      w.join();
    }
  }
  template<class T>
  void Push(const T& task) {
    std::function<void()> fnptr = task;
    task_queue_.Push(task);
  }

 private:
  DISALLOW_COPY_AND_ASSIGN(ThreadPool);
  ThreadPool();
  std::vector<std::thread> workers_;
  ConcurrentBlockingQueue<Task> task_queue_;

  void SimpleWorker(int thrid) {
    while (true) {
      Task task;
      bool exit_now = task_queue_.Pop(task);
      if (exit_now) {
        return;
      }
      task();
    }
  }
};

}

