#pragma once
#include <mutex>
#include <condition_variable>
#include <limits>
#include "minerva/common/common.h"

class SharedMutex {
 public:
  SharedMutex() = default;
  DISALLOW_COPY_AND_MOVE(SharedMutex);
  ~SharedMutex() = default;
  void Lock();
  void Unlock();
  void LockShared();
  void UnlockShared();

 private:
  std::mutex mutex_;
  std::condition_variable gate1_;
  std::condition_variable gate2_;
  uint64_t state_ = 0;

  static constexpr uint64_t writer_entered_ =
    std::numeric_limits<uint64_t>::max >> 1;
  static constexpr uint64_t reader_entered_ = ~writer_entered_;
};

