#include <dmlc/io.h>
#include <dmlc/logging.h>
#include "minerva.h"
#include "io/data.h"
#include "utils/config.h"

using namespace std;
using namespace minerva;

int main(int argc, char** argv) {
  MinervaSystem::Initialize(&argc, &argv);
  auto& ms = MinervaSystem::Instance();
  auto cpu_device = ms.device_manager().CreateCpuDevice();
  ms.SetDevice(cpu_device);
 
  DataProvider dp(argv[1]); 
  std::vector<NArray> next_batch = dp.GetNextValue();
  for (int i=0; i<100; i++)
  {
    std::vector<NArray> now_batch = next_batch;
    //data
    NArray label = now_batch[1];
    float* label_list = label.Get().get();
    for (int j = 0; j < label.Size()[1]; j++)
      std::cout << label_list[i] << " ";
    std::cout << "\n";
    next_batch = dp.GetNextValue();
  }
}
