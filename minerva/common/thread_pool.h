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
  typedef std::function<void(int)> Task;
  ThreadPool(size_t numthreads) : num_tasks_unfinished_(0) {
    for(size_t thrid = 0; thrid < numthreads; ++thrid) {
      workers_.emplace_back(&ThreadPool::SimpleWorker, this, thrid);
    }
  }
  ~ThreadPool() {
    WaitForAllFinished();
    task_queue_.SignalForKill();
    for (auto& w : workers_) {
      w.join();
    }
  }
  void Push(const Task& task) {
    ++num_tasks_unfinished_;
    task_queue_.Push(task);
  }
  void WaitForAllFinished() {
    while (num_tasks_unfinished_ != 0) {
      std::this_thread::yield();
    }
  }

 private:
  ThreadPool();
  std::vector<std::thread> workers_;
  ConcurrentBlockingQueue<Task> task_queue_;
  std::atomic<int> num_tasks_unfinished_;
  DISALLOW_COPY_AND_ASSIGN(ThreadPool);

  void SimpleWorker(int thrid) {
    Task task;
    while (!task_queue_.Pop(task)) {
      task(thrid);
      --num_tasks_unfinished_;
    }
  }
};

}  // namespace minerva

