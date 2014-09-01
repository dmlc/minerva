#pragma once
#include "common/singleton.h"
#include "device_info.h"
#include "device.h"

namespace Minerva {

class DeviceFactory : public EverlastingSingleton<DeviceFactory> {
 public:
  void Reset();
  int allocated() { return allocated_; }
  void PrintDevice(DeviceInfo device_info);
  DeviceInfo DefaultInfo();
  DeviceInfo GpuDeviceInfo(int gid);
  DeviceInfo GpuDeviceInfo(int gid, int num_stream);
  
 private:
  int allocated_;
  //map<uint64_t, Device> device_storage_;
};

}

