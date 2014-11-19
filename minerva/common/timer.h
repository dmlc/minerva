#include <sys/time.h>

class Timer {
 public:
  Timer() {
  }
  void Start() {
    if (started_) {
      return;
    }
    gettimeofday(&tv_, 0);
    started_ = true;
  }
  void Stop() {
    if (!started_) {
      return;
    }
    timeval end_tv;
    gettimeofday(&end_tv, 0);
    total_ += ((double) end_tv.tv_usec - tv_.tv_usec) / 1000000 + end_tv.tv_sec - tv_.tv_sec;
    started_ = false;
  }
  void Reset() {
    Stop();
    total_ = 0;
  }
  double Last() const {
    return total_;
  }

 private:
  bool started_ = false;
  timeval tv_;
  double total_ = 0;
};

