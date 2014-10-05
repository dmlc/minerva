#pragma once
#include <unordered_map>
#include <mutex>
#include "common/common.h"

template<typename K, typename V> class ConcurrentUnorderedMap {
 public:
  ConcurrentUnorderedMap() {
  }
  ~ConcurrentUnorderedMap() {
  }
  V& operator[](const K& k) {
    std::lock_guard<std::mutex> lck(m_);
    return map_[k];
  }
  size_t Erase(const K& k) {
    std::lock_guard<std::mutex> lck(m_);
    return map_.erase(k);
  }

 private:
  std::mutex m_;
  std::unordered_map<K, V> map_;
  DISALLOW_COPY_AND_ASSIGN(ConcurrentUnorderedMap);
};

