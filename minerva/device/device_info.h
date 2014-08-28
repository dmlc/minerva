#pragma once
#include <vector>
#include <cstring>
#include "common/singleton.h"

namespace minerva {

struct DeviceInfo {
  int id;
  std::vector<const char*> CPUList; // for now # of CPU should be 1, which is local
  std::vector<int> GPUList; // GPUs assigned to this "device", suitable for CUDA call cudaSetDevice
  std::vector<int> numStreams; // # of streams available on each GPU
};

class DeviceFactory : public EverlastingSingleton<DeviceFactory> {
  public:
    void Initialize();
    void print_device(DeviceInfo device_info);
    DeviceInfo default_info();
    DeviceInfo gpu_device_info(int gid);
    DeviceInfo gpu_device_info(int gid, int numStream);
  private:
    int allocated;
};

}
