#include "procedures/thread_pool.h"
#include <thread>
#include <functional>

using namespace std;

namespace minerva {

ThreadPool::ThreadPool(size_t size, std::function<void()> runner) {
  while (size--) {
    workers_.push_back(thread(runner));
  }
}

ThreadPool::~ThreadPool() {
  for (auto& i: workers_) {
    i.join();
  }
}

}
