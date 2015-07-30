#include <iomanip>
#include <dmlc/io.h>
#include <dmlc/logging.h>
#include "minerva.h"
#include "io/data.h"
#include "utils/config.h"
#include <time.h>

using namespace std;
using namespace minerva;

int main(int argc, char** argv) {
  MinervaSystem::Initialize(&argc, &argv);
  auto& ms = MinervaSystem::Instance();
  auto cpu_device = ms.device_manager().CreateCpuDevice();
  ms.SetDevice(cpu_device);

  DataProvider dp(argv[1]); 
  
  time_t start, end;
  time(&start);
  std::vector<NArray> next_batch = dp.GetNextValue();
  for (int i=0; i<100; i++)
  {
    std::cout << "batch " << i << std::endl;
    std::vector<NArray> now_batch = next_batch;
    //data
    NArray label = now_batch[1];
    NArray  data = now_batch[0];
    data.Wait();
    /*
    FileFormat f;
    f.binary = false; 
    label.ToStream(std::cout, f);
    std::cout << "\n";
    */
    next_batch = dp.GetNextValue();
  }
  time(&end);
  printf("The difference is: %f seconds", difftime(end, start));
}
