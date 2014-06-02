#include "procedures/thread_pool.h"
#include "common/concurrent_blocking_queue.h"
#include <thread>
#include <functional>

using namespace std;

namespace minerva {

ThreadPool::ThreadPool(size_t size) {
  while (size--) {
    workers_.push_back(thread(&ThreadPool::SimpleWorker, this));
  }
}

ThreadPool::~ThreadPool() {
  task_queue_.SignalForKill();
  for (auto& i: workers_) {
    i.join();
  }
}

void ThreadPool::AppendTask(Task t, Callback c) {
  task_queue_.Push(make_pair(t, c));
}

void ThreadPool::SimpleWorker() {
  while (true) {
    TaskPair task;
    bool exit_now = task_queue_.Pop(task);
    if (exit_now) {
      return;
    }
    task.first->runner()();
    task.second(task.first);
  }
}

}

