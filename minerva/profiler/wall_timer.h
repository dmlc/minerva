#pragma once
#include <sys/time.h>
#include "profiler/timer.h"

namespace minerva {

class WallTimer : public Timer {
 public:
  WallTimer();
  WallTimer(const WallTimer&);
  WallTimer& operator=(const WallTimer&);
  ~WallTimer();
  void Start();
  void Stop();
  double StartTimeMicrosecond() const;
  double EndTimeMicrosecond() const;
  double ReadMicrosecond() const;

 private:
  timeval start_;
  timeval end_;
};

}  // namespace minerva
