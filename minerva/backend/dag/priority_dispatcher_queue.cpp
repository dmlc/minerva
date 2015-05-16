#include "priority_dispatcher_queue.h"
#include <thread>
#include <dmlc/logging.h>

using namespace std;

namespace minerva {

PriorityDispatcherQueue::PriorityDispatcherQueue() :
  exit_now_(false),
  tasks_(static_cast<size_t>(TaskType::kEnd)),
  total_(0) {
}

void PriorityDispatcherQueue::Push(const TaskPair& task_pair) {
  unique_lock<mutex> lock(m_);
  tasks_.at(static_cast<size_t>(task_pair.first)).push_back(task_pair.second);
  if (++total_ == 1) {
    cv_.notify_all();
  }
}

bool PriorityDispatcherQueue::Pop(TaskPair& task_pair) {
  unique_lock<mutex> lock(m_);
  while (total_ == 0 && !exit_now_.Read()) {
    cv_.wait(lock);
  }
  if (exit_now_.Read()) {
    return true;
  } else {
    int available_index = -1;
    for (size_t i = 0; i < tasks_.size(); ++i) {
      if (!tasks_[i].empty()) {
        available_index = i;
        break;
      }
    }
    CHECK_NE(available_index, -1) << "empty task queue woken up";
    task_pair.first = static_cast<TaskType>(available_index);
    task_pair.second = tasks_[available_index].front();
    tasks_[available_index].pop_front();
    --total_;
    return false;
  }
}

void PriorityDispatcherQueue::SignalForKill() {
  unique_lock<mutex> lock(m_);
  exit_now_.Write(true);
  cv_.notify_all();
}

}  // namespace minerva

