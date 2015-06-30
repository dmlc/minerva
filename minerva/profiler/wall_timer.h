# pragma once

# if defined(_MSC_VER)
# include <winsock.h>
inline int gettimeofday(struct timeval* p, void* tz /* IGNORED */)
{
  union {
    long long ns100; /*time since 1 Jan 1601 in 100ns units */
     FILETIME ft;
  } now;

  GetSystemTimeAsFileTime(&(now.ft));
  p->tv_usec = (long)((now.ns100 / 10LL) % 1000000LL);
  p->tv_sec = (long)((now.ns100 - (116444736000000000LL)) / 10000000LL);
  return 0;
}

# else
#include <sys/time.h>
# endif

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