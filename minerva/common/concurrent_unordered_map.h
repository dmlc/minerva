#pragma once
#include <unordered_map>
#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>
#include "common/common.h"

template<typename K, typename V>
class ConcurrentUnorderedMap {
 public:
  ConcurrentUnorderedMap() = default;
  DISALLOW_COPY_AND_ASSIGN(ConcurrentUnorderedMap);
  ~ConcurrentUnorderedMap() = default;
  V& operator[](const K& k) {
    WriteLock lock(l_);
    return map_[k];
  }
  size_t Erase(const K& k) {
    WriteLock lock(l_);
    return map_.erase(k);
  }
  size_t Insert(const typename std::unordered_map<K, V>::value_type& v) {
    WriteLock lock(l_);
    return map_.insert(v).second;
  }
  V& At(const K& k) {
    ReadLock lock(l_);
    return map_.at(k);
  }
  const V& At(const K& k) const {
    ReadLock lock(l_);
    return map_.at(k);
  }
  size_t Size() const {
    ReadLock lock(l_);
    return map_.size();
  }
  void LockRead() const {
    l_.lock_shared();
  }
  void UnlockRead() const {
    l_.unlock_shared();
  }
  std::unordered_map<K, V>& VolatilePayload() {
    return map_;
  }
  const std::unordered_map<K, V>& VolatilePayload() const {
    return map_;
  }

 private:
  typedef boost::shared_mutex Lock;
  typedef boost::unique_lock<Lock> WriteLock;
  typedef boost::shared_lock<Lock> ReadLock;
  mutable Lock l_;
  std::unordered_map<K, V> map_;
};

