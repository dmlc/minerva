#pragma once

#include <map>
#include "common/singleton.h"

namespace minerva {

class DataStore : public Singleton<DataStore> {
 public:
  enum MemTypes {
    CPU = 0,
    GPU,
  };
  static uint64_t GenerateDataId();
  bool CreateData(uint64_t, MemTypes);
  float* GetData(uint64_t, MemTypes);
  void FreeData(uint64_t, MemTypes);
 private:
  static uint64_t data_id_gen_;
  std::map<uint64_t, float*> data_pointers_;
};

} // end of namespace minerva
