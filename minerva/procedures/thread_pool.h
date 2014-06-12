#pragma once
#include "common/common.h"
#include "common/concurrent_blocking_queue.h"
#include "dag/dag_node.h"
#include <vector>
#include <functional>
#include <thread>
#include <utility>

namespace minerva {

class DagEngine;

class ThreadPool {
 public:
  ThreadPool(size_t, DagEngine*);
  ~ThreadPool();

 private:
  DISALLOW_COPY_AND_ASSIGN(ThreadPool);
  ThreadPool();
  DagEngine* engine_;
  std::vector<std::thread> workers_;
  void SimpleWorker(DagEngine*);
};

}

