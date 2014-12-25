#include "profiler/cpu_timer.h"
#include <ctime>

namespace minerva {

CpuTimer::CpuTimer() : start_(0), end_(0) {
}

CpuTimer::CpuTimer(const CpuTimer& t) : start_(t.start_), end_(t.end_) {
}

CpuTimer& CpuTimer::operator=(const CpuTimer& t) {
  if (this == &t) {
    return *this;
  }
  start_ = t.start_;
  end_ = t.end_;
  return *this;
}

CpuTimer::~CpuTimer() {
}

void CpuTimer::Start() {
  start_ = clock();
}

void CpuTimer::Stop() {
  end_ = clock();
}

double CpuTimer::ReadMicrosecond() const {
  return ((double) end_ - start_) / CLOCKS_PER_SEC;
}

}  // namespace minerva

