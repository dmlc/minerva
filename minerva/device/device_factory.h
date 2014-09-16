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
  uint64_t CreateCPUDevice();
  uint64_t CreateGPUDevice(int gid);
  uint64_t CreateGPUDevice(int gid, int num_stream);
  Device* GetDevice(uint64_t id);
  
 private:
  void InsertCPUDevice(uint64_t id, DeviceInfo info);
  void InsertGPUDevice(uint64_t id, DeviceInfo info);

  uint64_t allocated_;
  std::unordered_map<uint64_t, Device*> device_storage_;
};

}

