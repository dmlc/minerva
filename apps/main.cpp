#include <minerva.h>
#include <iostream>
#include <cstdint>

using namespace std;
using namespace minerva;

int main(int argc, char** argv) {
  MinervaSystem::Initialize(&argc, &argv);
  auto& ms = MinervaSystem::Instance();
  auto gpu_device = ms.device_manager().CreateGpuDevice(0);
  ms.SetDevice(gpu_device);
  vector<NArray> narrs;
  for (int i = 0; i < 10000; ++i) {
    narrs.push_back(NArray::Constant({1, 1}, i));
  }
  cout << "a" << endl;
  for (int i = 0; i < static_cast<int>(narrs.size()); ++i) {
    narrs[i] = narrs[i] + 1;
  }
  cout << "b" << endl;
  for (int i = 0; i < static_cast<int>(narrs.size()); ++i) {
    narrs[i].Wait();
  }
  cout << "c" << endl;
  ms.backend().WaitForAll();
}
