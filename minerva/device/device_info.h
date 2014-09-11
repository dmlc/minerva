#pragma once
#include <cstdint>
#include <vector>

namespace minerva {

struct DeviceInfo {
  std::vector<const char*> cpu_list; // for now # of CPU should be 1, which is local
  std::vector<int> gpu_list; // GPUs assigned to this "device", suitable for CUDA call cudaSetDevice
  std::vector<int> num_streams; // # of streams available on each GPU
};

}
