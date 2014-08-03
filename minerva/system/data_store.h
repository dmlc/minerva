#pragma once
#include "common/common.h"
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
  bool CreateData(uint64_t, MemTypes, size_t len, int rc = 0);
  float* GetData(uint64_t, MemTypes);
  bool ExistData(uint64_t ) const;
  void IncrReferenceCount(uint64_t, int amount = 1);
  void DecrReferenceCount(uint64_t, int amount = 1);
  int GetReferenceCount(uint64_t ) const;
  //void FreeData(uint64_t, MemTypes);

 private:
  struct DataState {
    float* data_ptrs[NUM_MEM_TYPES];
    size_t length;
    int reference_count;
    DataState();
  };
  DISALLOW_COPY_AND_ASSIGN(DataStore);
  bool CheckValidity(uint64_t ) const;
  void GC(uint64_t );

 private:
  mutable std::mutex access_mutex_;
  std::map<uint64_t, DataState> data_states_;
};

inline bool DataStore::ExistData(uint64_t id) const {
  std::lock_guard<std::mutex> lck(access_mutex_);
  return data_states_.find(id) != data_states_.end();
}

inline DataStore::DataState::DataState(): length(0), reference_count(0) {
  // TODO use memset instead ?
  for(int i = 0; i < NUM_MEM_TYPES; ++i) data_ptrs[i] = NULL;
}

} // end of namespace minerva

