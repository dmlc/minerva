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
  int n = 3; // num samples
  int k = 3; // num features
  NArray a = NArray::Randn({m, k}, 0.0, 1.0, {1, 1});
  NArray b = NArray::Randn({k, n}, 0.0, 1.0, {1, 1});
  ms.set_device_id(gpuDevice);

  NArray c = a * b;

  ms.set_device_id(cpuDevice);
  FileFormat format; format.binary = false;
  a.ToFile("a.txt", format);
  b.ToFile("b.txt", format);
  c.ToFile("c.txt", format);
  return 0;
}
