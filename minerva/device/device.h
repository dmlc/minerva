#pragma once
#include <vector>
#include "device_info.h"
#include "system/data_store.h"

namespace minerva {

class cudaStream;

class Device {
 public:
  virtual void Execute(std::vector<DataShard> inputs, PhysicalOp Op); // called by Physical_Engine::ProcessNode()
  cudaStream GetStream();

 private:
  std::vector<uint64_t> local_data;
};

}
