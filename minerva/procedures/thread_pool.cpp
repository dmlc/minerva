#include "procedures/thread_pool.h"
#include "procedures/physical_engine.h"
#include <cstdio>
#include <thread>
#include <functional>

using namespace std;

namespace minerva {

ThreadPool::ThreadPool(size_t size, PhysicalEngine* engine): engine_(engine) {
  while (size--) {
    workers_.push_back(thread(&ThreadPool::SimpleWorker, this, engine_));
  }
}

ThreadPool::~ThreadPool() {
  for (auto& i: workers_) {
    i.join();
  }
}

void ThreadPool::SimpleWorker(PhysicalEngine* engine) {
  while (true) {
    PhysicalEngine::TaskPair task;
    bool exit_now = engine->GetNewTask(this_thread::get_id(), task);
    if (exit_now) {
      return;
    }
    task.second(task.first);
  }
}

}

