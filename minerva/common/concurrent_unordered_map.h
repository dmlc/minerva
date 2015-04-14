#pragma once
#include <unordered_map>
#include <memory>
#include "common/shared_mutex.h"
#include "common/common.h"

template<typename K, typename V>
class ConcurrentUnorderedMap {
 public:
  using Type = std::unordered_map<K, V>;

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
  size_t Insert(const typename Type::value_type& v) {
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
  std::shared_ptr<ReaderLock> GetReaderLock() const {
    return make_shared<ReaderLock>(m_);
  }
  Type& VolatilePayload() {
    return map_;
  }
  const Type& VolatilePayload() const {
    return map_;
  }

 private:
  using Mutex = minerva::common::SharedMutex;
  using ReaderLock = minerva::common::ReaderLock<Mutex>;
  using WriterLock = minerva::common::WriterLock<Mutex>;
  mutable Mutex m_;
  Type map_;
};

