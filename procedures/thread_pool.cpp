#include "procedures/thread_pool.h"
#include "procedures/dag_engine.h"
#include "common/concurrent_blocking_queue.h"
#include <cstdio>
#include <thread>
#include <functional>

using namespace std;

namespace minerva {

ThreadPool::ThreadPool(size_t size, DagEngine* engine): engine_(engine) {
  while (size--) {
    workers_.push_back(thread(&ThreadPool::SimpleWorker, this, engine_));
  }
}

ThreadPool::~ThreadPool() {
  for (auto& i: workers_) {
    i.join();
  }
}

void ThreadPool::SimpleWorker(DagEngine* engine) {
  while (true) {
    DagEngine::TaskPair task;
    bool exit_now = engine->GetNewTask(this_thread::get_id(), task);
    if (exit_now) {
      return;
    }
    printf("First start\n");
    task.first->runner()();
    printf("First complete\n");
    task.second(task.first);
    printf("Second complete\n");
  }
}

}

