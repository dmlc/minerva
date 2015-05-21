#pragma once
#include <unordered_set>
#include "common/common.h"
#include "common/shared_mutex.h"

template<typename T>
class ConcurrentUnorderedSet {
 public:
  ConcurrentUnorderedSet() = default;
  DISALLOW_COPY_AND_MOVE(ConcurrentUnorderedSet);
  ~ConcurrentUnorderedSet() = default;
  size_t Erase(const T& k) {
    WriterLock lock(m_);
    return set_.erase(k);
  }
  size_t Count(const T& k) {
    ReaderLock lock(m_);
    return set_.count(k);
  }
  bool Insert(const T& k) {
    WriterLock lock(m_);
    return set_.insert(k).second;
  }

 private:
  using Mutex = minerva::common::SharedMutex;
  using ReaderLock = minerva::common::ReaderLock<Mutex>;
  using WriterLock = minerva::common::WriterLock<Mutex>;
  mutable Mutex m_;
  std::unordered_set<T> set_;
};

