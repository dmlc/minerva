#include <minerva.h>
#include <iostream>
#include <cstdint>

using namespace std;
using namespace minerva;

int main(int argc, char** argv) {
  MinervaSystem::Initialize(&argc, &argv);
  auto& ms = MinervaSystem::Instance();
  auto gpu_device = ms.device_manager().CreateGpuDevice(0);
  ms.current_device_id_ = gpu_device;
  vector<NArray> narrs;
  for (int i = 0; i < 1000; ++i) {
    narrs.push_back(NArray::Constant({10, 10}, i));
  }
  for (int i = 0; i < narrs.size(); ++i) {
    narrs[i] = narrs[i] * 100 + 1;
  }
  for (int i = 0; i < narrs.size(); ++i) {
    narrs[i].Wait();
  }
  cout << ms.physical_dag().ToDotString() << endl;
  MinervaSystem::Finalize();
}
