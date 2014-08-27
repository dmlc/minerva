#pragma once
#include <vector>
#include "device_info.h"

namespace minerva {

class cudaStream;

class device {
 public:
  virtual void Execute(vector<DataShard> inputs, vector<DataShard> outputs, PhysicalOp Op); // called by Physical_Engine::ProcessNode()
  cudaStream GetStream();
};

}
