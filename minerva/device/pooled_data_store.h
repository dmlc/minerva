#pragma once
#include <queue>
#include "device/data_store.h"

namespace minerva {

class PooledDataStore : DataStore {
 public:
  PooledDataStore(size_t threshold, std::function<void*(size_t)> a, std::function<void(void*)> d);
  virtual ~PooledDataStore();
  virtual float* CreateData(uint64_t, size_t);
  virtual void FreeData(uint64_t);
  virtual size_t GetTotalBytes() const;

 private:
  size_t threshold_;
  size_t total_ = 0;
  std::unordered_map<size_t, std::queue<void*>> free_space_;
  void ReleaseFreeSpace();
  DISALLOW_COPY_AND_ASSIGN(PooledDataStore);
};

}  // namespace minerva

