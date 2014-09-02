#pragma once
#include <vector>
#include "device_info.h"
#include "system/data_store.h"

namespace minerva {

class Device {
 public:
  Device();
  Device(DeviceInfo info);
  DeviceInfo GetInfo();
  void Execute(std::vector<int> inputs); // called by Physical_Engine::ProcessNode()

 protected:
  std::vector<uint64_t> local_data_;
  DeviceInfo device_info_;
};

}
