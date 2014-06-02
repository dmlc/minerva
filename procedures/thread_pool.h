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
  typedef DagNode* Task;
  typedef std::function<void(DagNode*, ThreadPool*)> Callback;
  typedef std::pair<Task, Callback> TaskPair;
  ThreadPool(size_t);
  ~ThreadPool();
  void AppendTask(Task, Callback);

 private:
  DISALLOW_COPY_AND_ASSIGN(ThreadPool);
  ThreadPool();
  std::vector<std::thread> workers_;
  ConcurrentBlockingQueue<TaskPair> task_queue_;
  void SimpleWorker();
};

}

