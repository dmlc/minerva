#pragma once
#include <map>
#include "common/singleton.h"
#include "device_info.h"
#include "device.h"

namespace minerva {

class DeviceFactory : public EverlastingSingleton<DeviceFactory> {
 public:
  void Reset();
  int allocated() { return allocated_; }
  void PrintDevice(DeviceInfo device_info);
  DeviceInfo DefaultInfo();
  DeviceInfo GPUDeviceInfo(int gid);
  DeviceInfo GPUDeviceInfo(int gid, int num_stream);
  Device GetDevice(uint64_t id);
  Device GetDevice(DeviceInfo info);
  
 private:
  void InsertGPUDevice(DeviceInfo info);

  uint64_t allocated_;
  std::map<uint64_t, Device> device_storage_;
};

}

