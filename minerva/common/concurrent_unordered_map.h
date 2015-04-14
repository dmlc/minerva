#pragma once
#include <unordered_map>
#include "common/shared_mutex.h"
#include "common/common.h"

template<typename K, typename V>
class ConcurrentUnorderedMap {
 public:
  ConcurrentUnorderedMap() = default;
  DISALLOW_COPY_AND_MOVE(ConcurrentUnorderedMap);
  ~ConcurrentUnorderedMap() = default;
  V& operator[](const K& k) {
    WriterLock lock(l_);
    return map_[k];
  }
  size_t Erase(const K& k) {
    WriterLock lock(l_);
    return map_.erase(k);
  }
  size_t Insert(const typename std::unordered_map<K, V>::value_type& v) {
    WriterLock lock(l_);
    return map_.insert(v).second;
  }
  V& At(const K& k) {
    ReaderLock lock(l_);
    return map_.at(k);
  }
  const V& At(const K& k) const {
    ReaderLock lock(l_);
    return map_.at(k);
  }
  size_t Size() const {
    ReaderLock lock(l_);
    return map_.size();
  }
  void LockRead() const {
    l_.LockShared();
  }
  void UnlockRead() const {
    l_.UnlockShared();
  }
  std::unordered_map<K, V>& VolatilePayload() {
    return map_;
  }
  const std::unordered_map<K, V>& VolatilePayload() const {
    return map_;
  }

 private:
  using Mutex = minerva::common::SharedMutex;
  using ReaderLock = minerva::common::ReaderLock<Mutex>;
  using WriterLock = minerva::common::WriterLock<Mutex>;
  mutable Mutex m_;
  std::unordered_map<K, V> map_;
};

