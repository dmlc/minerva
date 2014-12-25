#include "profiler/execution_profiler.h"

using namespace std;

namespace minerva {

ExecutionProfiler::ExecutionProfiler() {
}

ExecutionProfiler::~ExecutionProfiler() {
}

void ExecutionProfiler::RecordTime(TimerType type, const string& name, const Timer& timer) {
  lock_guard<mutex> lock_(m_);
  auto it = time_.find(name);
  if (it == time_.end()) {  // Not existent before
    time_.insert({name, new double[static_cast<size_t>(TimerType::kCount)]()});
  } else {
    (it->second)[static_cast<size_t>(type)] += timer.ReadMicrosecond();
  }
}

}  // namespace minerva
