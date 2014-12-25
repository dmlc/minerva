#include <glog/logging.h>
#include "profiler/timer.h"

namespace minerva {

Timer::Timer() {
  start_.tv_sec = start_.tv_usec = 0;
  end_.tv_sec = end_.tv_usec = 0;
}

Timer::Timer(const Timer& t) : start_(t.start_), end_(t.end_) {
}

Timer& Timer::operator=(const Timer& t) {
  if (this == &t) {
    return *this;
  }
  start_ = t.start_;
  end_ = t.end_;
  return *this;
}

void Timer::Start() {
  if (gettimeofday(&start_, 0)) {
    LOG(FATAL) << "timer handle error";
  }
}

void Timer::Stop() {
  if (gettimeofday(&end_, 0)) {
    LOG(FATAL) << "timer handle error";
  }
}

double Timer::ReadMicrosecond() {
  return (end_.tv_sec - start_.tv_sec) * 1000 + (end_.tv_usec - start_.tv_usec) * .001;
}

}  // namespace minerva

