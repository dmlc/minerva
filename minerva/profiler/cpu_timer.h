#pragma once
#include <ctime>
#include "profiler/timer.h"

namespace minerva {

class CpuTimer : public Timer {
 public:
  CpuTimer();
  CpuTimer(const CpuTimer&);
  CpuTimer& operator=(const CpuTimer&);
  ~CpuTimer();
  void Start();
  void Stop();
  double StartTimeMicrosecond() const;
  double EndTimeMicrosecond() const;
  double ReadMicrosecond() const;

 private:
  clock_t start_;
  clock_t end_;
};

}  // namespace minerva
