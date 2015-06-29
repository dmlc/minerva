#pragma once
#include <mutex>
#include <condition_variable>
#include <limits>
#include "common/common.h"

namespace minerva {
namespace common {

template<typename Mutex>
class ReaderLock {
 public:
  static void Lock(Mutex& m_) {
    m_.LockShared();
  }
  static void Unlock(Mutex& m_) {
    m_.UnlockShared();
  }
  explicit ReaderLock(Mutex& m) : mutex_(&m) {
    mutex_->LockShared();
  }
  DISALLOW_COPY_AND_MOVE(ReaderLock);
  ~ReaderLock() {
    mutex_->UnlockShared();
  }

 private:
  Mutex* mutex_;
};

template<typename Mutex>
class WriterLock {
 public:
  static void Lock(Mutex& m_) {
    m_.Lock();
  }
  static void Unlock(Mutex& m_) {
    m_.Unlock();
  }
  explicit WriterLock(Mutex& m) : mutex_(&m) {
    mutex_->Lock();
  }
  DISALLOW_COPY_AND_MOVE(WriterLock);
  ~WriterLock() {
    mutex_->Unlock();
  }

 private:
  Mutex* mutex_;
};

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

# if defined (_MSC_VER)
  static const uint64_t reader_entered_ =
    _ULLONG_MAX >> 1;
  static const uint64_t writer_entered_ = ~reader_entered_;
# else
  static constexpr uint64_t reader_entered_ =
    std::numeric_limits<uint64_t>::max() >> 1;
  static constexpr uint64_t writer_entered_ = ~reader_entered_;
# endif
};

}  // namespace common
}  // namespace minerva

