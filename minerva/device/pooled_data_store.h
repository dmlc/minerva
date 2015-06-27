#pragma once
#include <queue>
#include "device/data_store.h"

namespace minerva {

class PooledDataStore final : public DataStore {
 public:
  PooledDataStore(size_t threshold, std::function<void*(size_t)> a, std::function<void(void*)> d);
  DISALLOW_COPY_AND_ASSIGN(PooledDataStore);
  virtual ~PooledDataStore();
  float* CreateData(uint64_t, size_t) override;
  void FreeData(uint64_t) override;
  size_t GetTotalBytes() const override;
  std::unique_ptr<TemporarySpaceHolder> GetTemporarySpace(size_t) override;

 private:
  size_t threshold_;
  size_t total_ = 0;
  std::unordered_map<size_t, std::queue<void*>> free_space_;
  void ReleaseFreeSpace();
  virtual void FreeTemporarySpace(uint64_t) override;
};

}  // namespace minerva

