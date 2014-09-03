#pragma once
#include <vector>
#include "device_info.h"
#include "system/data_store.h"
#include "op/physical.h"

namespace minerva {

class Device {
 enum DeviceTypes {
  GPU_DEVICE = 0,
  CPU_DEVICE
 };
 public:
  Device();
  Device(DeviceInfo info);
  DeviceInfo GetInfo();
  void Execute(std::vector<PhysicalData> inputs, const PhysicalOp Op); // called by Physical_Engine::ProcessNode()

 protected:
  std::vector<uint64_t> local_data_;
  DeviceInfo device_info_;
};

}
