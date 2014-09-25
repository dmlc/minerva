#include <cstring>
#include <iostream>
#include <fstream>
#include "minerva.h"

using namespace std;
using namespace minerva;

int main(int argc, char** argv) {
  MinervaSystem& ms = MinervaSystem::Instance();
  ms.Initialize(&argc, &argv);
  {
    uint64_t gpuDevice = ms.CreateGpuDevice(0);
    uint64_t cpuDevice = ms.CreateCpuDevice();
    ms.current_device_id_ = cpuDevice;
    int m = 3;
    int k = 2;
    NArray a = NArray::Randn({m, k}, 0.0, 1.0);
    ms.current_device_id_ = gpuDevice;
    NArray b = 1 - a;
    ms.current_device_id_ = cpuDevice;
    FileFormat format{false};
    a.ToFile("a.txt", format);
    b.ToFile("b.txt", format);
  }
  ms.Finalize();
  return 0;
}
