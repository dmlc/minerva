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
  auto a = NArray::Randn({3, 2}, 0, 1);
  auto b = a.MaxIndex(0);
  auto top_diff_ptr = a.Get();
  for (int i = 0; i < a.Size().Prod(); ++i) {
    cout << top_diff_ptr.get()[i] << ' ';
  }
  cout << endl;
  top_diff_ptr = b.Get();
  for (int i = 0; i < b.Size().Prod(); ++i) {
    cout << top_diff_ptr.get()[i] << ' ';
  }
  cout << endl;
}

int main(int argc, char** argv) {
  auto& ms = MinervaSystem::Instance();
  ms.Initialize(&argc, &argv);
  cpu_device = ms.CreateCpuDevice();
  gpu_device = ms.CreateGpuDevice(1);
  Train();
  ms.dag_scheduler().GCNodes();
  cout << ms.device_manager().GetDevice(cpu_device)->GetMemUsage() << endl;
  cout << ms.device_manager().GetDevice(gpu_device)->GetMemUsage() << endl;
  ms.Finalize();
  return 0;
}
