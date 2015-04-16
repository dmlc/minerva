#pragma once
#include <unordered_map>
#include "common/shared_mutex.h"
#include "common/common.h"

template<typename K, typename V> class ConcurrentUnorderedMap {
 public:
  ConcurrentUnorderedMap() = default;
  DISALLOW_COPY_AND_ASSIGN(ConcurrentUnorderedMap);
  ~ConcurrentUnorderedMap() = default;
  V& operator[](const K& k) {
    WriterLock lock(m_);
    return map_[k];
  }
  size_t Erase(const K& k) {
    WriterLock lock(m_);
    return map_.erase(k);
  }
  size_t Insert(const typename std::unordered_map<K, V>::value_type& v) {
    WriterLock lock(m_);
    return map_.insert(v).second;
  }
  V& At(const K& k) {
    ReaderLock lock(m_);
    return map_.at(k);
  }
  const V& At(const K& k) const {
    ReaderLock lock(m_);
    return map_.at(k);
  }
  size_t Size() const {
    ReaderLock lock(m_);
    return map_.size();
  }
  void LockRead() const {
    ReaderLock::Lock(m_);
  }
  void UnlockRead() const {
    ReaderLock::Unlock(m_);
  }
  std::unordered_map<K, V>& VolatilePayload() {
    return map_;
  }
  const std::unordered_map<K, V>& VolatilePayload() const {
    return map_;
  }

 private:
  typedef minerva::common::SharedMutex Mutex;
  typedef minerva::common::ReaderLock<Mutex> ReaderLock;
  typedef minerva::common::WriterLock<Mutex> WriterLock;
  mutable Mutex m_;
  std::unordered_map<K, V> map_;
};

