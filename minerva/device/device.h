#pragma once
#include <vector>
#include "device_info.h"

using namespace std;

namespace minerva {

class cudaStream;

class Device {
 public:
  virtual void Execute(vector<DataShard> inputs, PhysicalOp Op); // called by Physical_Engine::ProcessNode()
  cudaStream GetStream();

 private:
  vector<uint64_t> local_data;
};

}
