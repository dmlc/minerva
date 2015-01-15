#include <glog/logging.h>
#include "profiler/wall_timer.h"

namespace minerva {

WallTimer::WallTimer() {
  start_.tv_sec = start_.tv_usec = 0;
  end_.tv_sec = end_.tv_usec = 0;
}

WallTimer::WallTimer(const WallTimer& t) : start_(t.start_), end_(t.end_) {
}

WallTimer& WallTimer::operator=(const WallTimer& t) {
  if (this == &t) {
    return *this;
  }
  start_ = t.start_;
  end_ = t.end_;
  return *this;
}

WallTimer::~WallTimer() {
}

void WallTimer::Start() {
  if (gettimeofday(&start_, 0)) {
    LOG(FATAL) << "timer handle error";
  }
}

void WallTimer::Stop() {
  if (gettimeofday(&end_, 0)) {
    LOG(FATAL) << "timer handle error";
  }
}

double WallTimer::ReadMicrosecond() const {
  return (end_.tv_sec - start_.tv_sec) * 1000 + (end_.tv_usec - start_.tv_usec) * .001;
}

}  // namespace minerva

