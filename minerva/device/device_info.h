#pragma once
#include "common/singleton.h"
#include <vector>
#include <cstring>

namespace minerva {

struct DeviceInfo {
  int id;
  std::vector<const char*> cpu_list; // for now # of CPU should be 1, which is local
  std::vector<int> gpu_list; // GPUs assigned to this "device", suitable for CUDA call cudaSetDevice
  std::vector<int> num_streams; // # of streams available on each GPU
};

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
};

}
