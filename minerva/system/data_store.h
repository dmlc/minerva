#pragma once
#include "common/common.h"
#include <unordered_map>
#include <map>
#include <cstdint>
#include <cstddef>
#include <mutex>

namespace minerva {

class DataStore {
 public:
  enum MemTypes {
    CPU = 0,
    GPU,
    NUM_MEM_TYPES
  };
  DataStore();
  ~DataStore();
  uint64_t GenerateDataID();
  void CreateData(uint64_t, MemTypes, size_t len, int rc = 0);
  float* GetData(uint64_t, MemTypes);
  bool ExistData(uint64_t) const;
  void FreeData(uint64_t);
  // return true if the RC is zero afterwards
  bool IncrReferenceCount(uint64_t, int amount = 1);
  bool DecrReferenceCount(uint64_t, int amount = 1);
  bool SetReferenceCount(uint64_t, int);
  int GetReferenceCount(uint64_t) const;
  size_t GetTotalBytes(MemTypes memtype) const;

 private:
  DISALLOW_COPY_AND_ASSIGN(DataStore);
  struct DataState {
    DataState();
    void* data_ptrs[NUM_MEM_TYPES];
    size_t length;
    int reference_count;
  };
  bool CheckValidity(uint64_t) const;
  void GC(uint64_t);
  mutable std::mutex access_mutex_;
  std::unordered_map<uint64_t, DataState> data_states_;
};

inline bool DataStore::ExistData(uint64_t id) const {
  std::lock_guard<std::mutex> lck(access_mutex_);
  return data_states_.find(id) != data_states_.end();
}

} // end of namespace minerva

