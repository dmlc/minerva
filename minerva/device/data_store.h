#pragma once
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <mutex>
#include <cstdint>
#include <cstddef>
#include <glog/logging.h>
#include "common/common.h"

namespace minerva {

class DataStore {
 public:
  DataStore(std::function<void*(size_t)> a, std::function<void(void*)> d);
  ~DataStore();
  float* CreateData(uint64_t, size_t);
  float* GetData(uint64_t);
  bool ExistData(uint64_t id) const;
  void FreeData(uint64_t);
  size_t GetTotalBytes() const;

 private:
  struct DataState {
    void* ptr;
    size_t length;
  };
  mutable std::mutex access_mutex_;
  std::unordered_map<uint64_t, DataState> data_states_;
  std::function<void*(size_t)> allocator_;
  std::function<void(void*)> deallocator_;
  DISALLOW_COPY_AND_ASSIGN(DataStore);
};

}  // namespace minerva

