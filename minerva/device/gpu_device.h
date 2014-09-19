#pragma once
#include <vector>
#include "device.h"

namespace minerva {

class GpuDevice : public Device {
 public:
  GpuDevice();
  GpuDevice(uint64_t id, DeviceInfo info);
  DeviceTypes Type() const { return GPU_DEVICE; }
  void CreateData(uint64_t data_id, int size);
  float* GetData(uint64_t data_id);
  void Execute_Op(std::vector<DataShard> inputShards, std::vector<DataShard> outputShards, PhysicalOp Op);
};

}

