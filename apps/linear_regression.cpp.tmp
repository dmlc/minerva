#include "minerva.h"

#include <cstring>
#include <fstream>

using namespace std;
using namespace minerva;

int main(int argc, char** argv) {
  MinervaSystem & ms = MinervaSystem::Instance();
  ms.Initialize(&argc, &argv);
  uint64_t cpuDevice = ms.CreateCPUDevice();
  uint64_t gpuDevice = ms.CreateGPUDevice(0);
  ms.set_device_id(cpuDevice);
  
  int n = 10; // num samples
  int k = 8; // num features
  NArray x = NArray::Randn({n, k}, 0.0, 1.0, {1, 1});
  NArray y = NArray::Randn({n, k}, 0.0, 1.0, {1, 1});
  NArray theta = NArray::Randn({k, k}, 0.0, 1.0, {1, 1});
  float alpha = 0.5; // lr
  int epoch = 2;

  ms.set_device_id(gpuDevice);

  FileFormat format; format.binary = false;
  for(int i = 0; i < epoch; ++i) {
    NArray error = x * theta - y;
    error.ToFile("error" + to_string(i), format);
    NArray alphaxt = alpha * x.Trans();
    alphaxt.ToFile("alphaxt" + to_string(i), format);
    NArray alphaxterror = alphaxt * error;
    alphaxterror.ToFile("alphaxterror" + to_string(i), format);
    theta = theta - alpha * x.Trans() * error;
    theta.ToFile("theta" + to_string(i), format);
  }

  ofstream fout_ldag("ldag.txt");
  fout_ldag << MinervaSystem::Instance().logical_dag().PrintDag() << endl;
  fout_ldag.close();
  
  theta.ToFile("theta.txt", format);
  //theta.Eval();
  ofstream fout_pdag("pdag.txt");
  fout_pdag << MinervaSystem::Instance().physical_dag().PrintDag<DataIdPrinter>() << endl;
  fout_pdag.close();
  return 0;
}
