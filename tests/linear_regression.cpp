#include "minerva.h"

#include <cstring>
#include <fstream>

using namespace std;
using namespace minerva;

int main() {
  int n = 10; // num samples
  int k = 8; // num features
  NArray x = NArray::Randn({n, k}, 0.0, 1.0, {2, 1});
  NArray y = NArray::Randn({n, k}, 0.0, 1.0, {2, 1});
  NArray theta = NArray::Randn({k, k}, 0.0, 1.0, {1, 1});
  float alpha = 0.5; // lr
  int epoch = 2;
  for(int i = 0; i < epoch; ++i) {
    NArray error = x * theta - y;
    theta = theta - alpha * x.Trans() * error;
  }
  ofstream fout_ldag("ldag.txt");
  fout_ldag << MinervaSystem::Instance().logical_dag().PrintDag() << endl;
  fout_ldag.close();
  theta.Eval();
  ofstream fout_pdag("pdag.txt");
  fout_pdag << MinervaSystem::Instance().physical_dag().PrintDag() << endl;
  fout_pdag.close();
  return 0;
}
