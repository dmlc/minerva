#include <minerva.h>

using namespace std;
using namespace minerva;

int main(int argc, char** argv) {
  MinervaSystem& ms = MinervaSystem::Instance();
  ms.Initialize(&argc, &argv);

  uint64_t cpu_device = ms.CreateCpuDevice();
  uint64_t gpu_device = ms.CreateGpuDevice(0);


