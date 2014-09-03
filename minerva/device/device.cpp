#include <iostream>
#include "device.h"

namespace minerva {

Device::Device() {}

Device::Device(DeviceInfo info) {
  device_info_ = info;
}

DeviceInfo Device::GetInfo() {
  return device_info_;
}

void Device::Execute(std::vector<PhysicalData> inputs, const PhysicalOp Op) {
  std::cout << Op.impl_type << std::endl;
}

}
