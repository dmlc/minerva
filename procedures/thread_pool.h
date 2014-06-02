#pragma once
#include "common/common.h"
#include "common/concurrent_blocking_queue.h"
#include "dag/dag_node.h"
#include <vector>
#include <functional>
#include <thread>
#include <utility>

namespace minerva {

class ThreadPool {
 public:
  typedef std::pair<DagNode*, std::function<void()>> Task;
  ThreadPool(size_t);
  ~ThreadPool();
  void AppendTask(Task);

 private:
  DISALLOW_COPY_AND_ASSIGN(ThreadPool);
  ThreadPool();
  std::vector<std::thread> workers_;
  ConcurrentBlockingQueue<Task> task_queue_;
  void SimpleWorker();
};

}

