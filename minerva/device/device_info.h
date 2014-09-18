#pragma once
#include <cstdint>
#include <vector>
#include <string>

namespace minerva {

struct DeviceInfo {
  std::vector<std::string> cpu_list; // For now # of CPU should be 1, which is local
  std::vector<int> gpu_list; // GPUs assigned to this "device", suitable for CUDA call cudaSetDevice
  std::vector<int> num_streams; // # of streams available on each GPU
};

}

