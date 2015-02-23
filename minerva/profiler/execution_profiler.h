#pragma once
#include <string>
#include <unordered_map>
#include <mutex>
#include "common/common.h"
#include "profiler/timer.h"

namespace minerva {

enum class TimerType {
  kMemory,
  kCalculation,
  kCount,
  kEnd
};

class ExecutionProfiler {
 public:
  ExecutionProfiler();
  virtual ~ExecutionProfiler();
  void RecordTime(TimerType, const std::string&, const Timer&);
  void Reset();
  void PrintResult();

 private:
  std::mutex m_;
  std::unordered_map<std::string, double*> time_;
  DISALLOW_COPY_AND_ASSIGN(ExecutionProfiler);
};

}  // namespace minerva
