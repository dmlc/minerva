#pragma once
#include "common/common.h"
#include "common/concurrent_blocking_queue.h"
#include <vector>
#include <functional>
#include <thread>
#include <utility>

namespace minerva {

class PhysicalEngine;

class ThreadPool {
 public:
  ThreadPool(size_t, PhysicalEngine*);
  ~ThreadPool();

 private:
  DISALLOW_COPY_AND_ASSIGN(ThreadPool);
  ThreadPool();
  PhysicalEngine* engine_;
  std::vector<std::thread> workers_;
  void SimpleWorker(PhysicalEngine*);
};

}

