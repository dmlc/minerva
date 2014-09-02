#include "device.h"

namespace minerva {

Device::Device() {}

Device::Device(DeviceInfo info) {
  device_info_ = info;
}

DeviceInfo Device::GetInfo() {
  return device_info_;
}

}
