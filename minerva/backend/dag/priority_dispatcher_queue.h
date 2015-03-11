#pragma once
#include <mutex>
#include <condition_variable>
#include <vector>
#include <list>
#include <utility>
#include "backend/dag/task_type.h"
#include "common/common.h"
#include "common/bool_flag.h"

namespace minerva {

class PriorityDispatcherQueue {
 public:
  typedef std::pair<TaskType, uint64_t> TaskPair;
  PriorityDispatcherQueue();
  DISALLOW_COPY_AND_ASSIGN(PriorityDispatcherQueue);
  ~PriorityDispatcherQueue() = default;
  void Push(const TaskPair&);
  bool Pop(TaskPair&);
  void SignalForKill();

 private:
  std::mutex m_;
  std::condition_variable cv_;
  BoolFlag exit_now_;
  std::vector<std::list<uint64_t>> tasks_;
};

}  // namespace minerva

