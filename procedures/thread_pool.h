#pragma once
#include "common/common.h"
#include <vector>
#include <functional>
#include <thread>

namespace minerva {

class ThreadPool {
 public:
  ThreadPool(size_t, std::function<void()>);
  ~ThreadPool();

 private:
  DISALLOW_COPY_AND_ASSIGN(ThreadPool);
  ThreadPool();
  std::vector<std::thread> workers_;
};

}

