#pragma once
#include <atomic>
#include "common/common.h"

namespace minerva {
namespace common {

class SpinLock {
 public:
  SpinLock() = default;
  DISALLOW_COPY_AND_MOVE(SpinLock);
  ~SpinLock() = default;
  void Lock();
  void Unlock();

 private:
  std::atomic_flag m_ = ATOMIC_FLAG_INIT;
};

}  // namespace common
}  // namespace minerva

