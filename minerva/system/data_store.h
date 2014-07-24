#pragma once
#include "common/common.h"
#include <map>
#include <cstdint>
#include <cstddef>

namespace minerva {

class DataStore {
 public:
  enum MemTypes {
    CPU = 0,
    GPU,
  };
  DataStore();
  ~DataStore();
  uint64_t GenerateDataID();
  bool CreateData(uint64_t, MemTypes, size_t);
  float* GetData(uint64_t, MemTypes);
  void FreeData(uint64_t, MemTypes);

 private:
  DISALLOW_COPY_AND_ASSIGN(DataStore);
  std::map<uint64_t, float*> data_pointers_;
};

} // end of namespace minerva

