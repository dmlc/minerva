#include "minerva.h"

#include <cstring>
#include <fstream>

using namespace std;
using namespace minerva;

int main(int argc, char** argv) {
  MinervaSystem & ms = MinervaSystem::Instance();
  ms.Initialize(&argc, &argv);
  uint64_t gpuDevice = ms.CreateGPUDevice(0);
  uint64_t cpuDevice = ms.CreateCPUDevice();
  ms.set_device_id(cpuDevice);
 
  int m = 3; 
  int k = 2; // num features
  NArray a = NArray::Randn({m, k}, 0.0, 1.0, {1, 1});
  ms.set_device_id(gpuDevice);

  NArray b = 1 - a;

  ms.set_device_id(cpuDevice);
  FileFormat format; format.binary = false;
  a.ToFile("a.txt", format);
  b.ToFile("b.txt", format);
  return 0;
}
