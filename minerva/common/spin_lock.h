#pragma once
#include <atomic>
#include "common/common.h"

namespace minerva {
namespace common {

class SpinLock {
 public:
# if defined(_MSC_VER)
  SpinLock() { m_.clear(); }
# else
  SpinLock() = default;
# endif
  DISALLOW_COPY_AND_MOVE(SpinLock);
  ~SpinLock() = default;
  void Lock();
  void Unlock();

 private:
# if defined(_MSC_VER)
   std::atomic_flag m_;
# else
   std::atomic_flag m_ = ATOMIC_FLAG_INIT;
# endif
};

}  // namespace common
}  // namespace minerva

