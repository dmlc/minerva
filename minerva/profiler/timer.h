#pragma once
#include <sys/time.h>

namespace minerva {

class Timer {
 public:
  Timer();
  Timer(const Timer&);
  Timer& operator=(const Timer&);
  virtual ~Timer();
  void Start();
  void Stop();
  double ReadMicrosecond();

 private:
  timeval start_;
  timeval end_;
};

}  // namespace minerva

