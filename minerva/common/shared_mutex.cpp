#include "shared_mutex.h"

using namespace std;

namespace minerva {
namespace common {

void SharedMutex::Lock() {
  unique_lock<mutex> lock(mutex_);
  while (state_ & writer_entered_) {
    gate1_.wait(lock);
  }
  state_ |= writer_entered_;
  while (state_ & reader_entered_) {
    gate2_.wait(lock);
  }
}

void SharedMutex::Unlock() {
  unique_lock<mutex> lock(mutex_);
  state_ = 0;
  gate1_.notify_all();
}

void SharedMutex::LockShared() {
  unique_lock<mutex> lock(mutex_);
  while ((state_ & writer_entered_) ||
      (state_ & reader_entered_) == reader_entered_) {
    gate1_.wait(lock);
  }
  uint64_t num_readers = (state_ & reader_entered_) + 1;
  state_ &= ~reader_entered_;
  state_ |= num_readers;
}

void SharedMutex::UnlockShared() {
  unique_lock<mutex> lock(mutex_);
  uint64_t num_readers = (state_ & reader_entered_) - 1;
  state_ &= ~reader_entered_;
  state_ |= num_readers;
  if (state_ & writer_entered_) {
    if (num_readers == 0) {
      gate2_.notify_one();
    }
  } else {
    if (num_readers == reader_entered_ - 1) {
      gate1_.notify_all();
    }
  }
}

}  // namespace common
}  // namespace minerva
