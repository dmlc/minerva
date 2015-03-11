#include "priority_dispatcher_queue.h"
#include <thread>

using namespace std;

namespace minerva {

PriorityDispatcherQueue::PriorityDispatcherQueue() : exit_now_(false), tasks_(static_cast<size_t>(TaskType::kEnd)) {
}

void PriorityDispatcherQueue::Push(const TaskPair& task_pair) {
  unique_lock<mutex> lock(m_);
  tasks_.at(static_cast<size_t>(task_pair.first)).push_back(task_pair.second);
}

bool PriorityDispatcherQueue::Pop(TaskPair& task_pair) {
  unique_lock<mutex> lock(m_);
  int available_index = -1;
  for (size_t i = 0; i < tasks_.size(); ++i) {
    if (!tasks_[i].empty()) {
      available_index = i;
      break;
    }
  }
  while (available_index == -1 && !exit_now_.Read()) {
    cv_.wait(lock);
    for (size_t i = 0; i < tasks_.size(); ++i) {
      if (!tasks_[i].empty()) {
        available_index = i;
        break;
      }
    }
  }
  if (exit_now_.Read()) {
    return true;
  } else {
    task_pair.first = static_cast<TaskType>(available_index);
    task_pair.second = tasks_[available_index].front();
    tasks_[available_index].pop_front();
    return false;
  }
}

void PriorityDispatcherQueue::SignalForKill() {
  unique_lock<mutex> lock(m_);
  exit_now_.Write(true);
  cv_.notify_all();
}

}  // namespace minerva

