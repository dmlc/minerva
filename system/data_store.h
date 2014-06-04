#pragma once
#include "common/singleton.h"
#include "common/common.h"
#include <map>
#include <cstdint>

namespace minerva {

class DataStore : public Singleton<DataStore> {
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
  uint64_t data_id_gen_;
  std::map<uint64_t, float*> data_pointers_;
};

} // end of namespace minerva

