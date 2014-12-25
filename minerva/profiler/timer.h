#pragma once

namespace minerva {

class Timer {
 public:
  Timer();
  virtual ~Timer();
  virtual void Start() = 0;
  virtual void Stop() = 0;
  virtual double ReadMicrosecond() const = 0;
};

}  // namespace minerva

