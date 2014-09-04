#pragma once
#include <unordered_map>
#include "common/singleton.h"
#include "device_info.h"
#include "device.h"

namespace minerva {

class DeviceFactory : public EverlastingSingleton<DeviceFactory> {
 public:
  DeviceFactory();
  void Reset();
  int allocated() { return allocated_; }
  void PrintDevice(DeviceInfo device_info);
  DeviceInfo DefaultInfo();
  DeviceInfo CreateGPUDevice(int gid);
  DeviceInfo CreateGPUDevice(int gid, int num_stream);
  Device* GetDevice(uint64_t id);
  Device* GetDevice(DeviceInfo info);
  
 private:
  void InsertCPUDevice(DeviceInfo info);
  void InsertGPUDevice(DeviceInfo info);

  uint64_t allocated_;
  std::unordered_map<uint64_t, Device*> device_storage_;
};

}

