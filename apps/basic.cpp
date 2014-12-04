#include <memory>
#include <iostream>
#include <minerva.h>
#include <cublas_v2.h>

using namespace std;
using namespace minerva;

uint64_t cpu_device, gpu_device;

void Train() {
  auto& ms = MinervaSystem::Instance();
  ms.current_device_id_ = gpu_device;
  auto a = NArray::Randn({1556925644}, 0, 1);
  a.WaitForEval();
}

int main(int argc, char** argv) {
  auto& ms = MinervaSystem::Instance();
  ms.Initialize(&argc, &argv);
  cpu_device = ms.CreateCpuDevice();
  gpu_device = ms.CreateGpuDevice(0);
  Train();
  ms.dag_scheduler().GCNodes();
  cout << ms.device_manager().GetDevice(cpu_device)->GetMemUsage() << endl;
  cout << ms.device_manager().GetDevice(gpu_device)->GetMemUsage() << endl;
  ms.Finalize();
  return 0;
}
