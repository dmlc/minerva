#pragma once
#include <unordered_set>
#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>
#include "common/common.h"

template<typename T>
class ConcurrentUnorderedSet {
 public:
  ConcurrentUnorderedSet() = default;
  DISALLOW_COPY_AND_ASSIGN(ConcurrentUnorderedSet);
  ~ConcurrentUnorderedSet() = default;
  size_t Erase(const T& k) {
    WriteLock lock(l_);
    return set_.erase(k);
  }
  size_t Count(const T& k) {
    ReadLock lock(l_);
    return set_.count(k);
  }
  bool Insert(const T& k) {
    WriteLock lock(l_);
    return set_.insert(k).second;
  }

 private:
  typedef boost::shared_mutex Lock;
  typedef boost::unique_lock<Lock> WriteLock;
  typedef boost::shared_lock<Lock> ReadLock;
  Lock l_;
  std::unordered_set<T> set_;
};

