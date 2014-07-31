#include "minerva.h"

#include <cstring>
#include <fstream>

using namespace std;
using namespace minerva;

void Print(float* ptr) {
  for(int i = 0; i < 10; ++i)
    cout << ptr[i] << " ";
  cout << endl;
}

int main(int argc, char** argv) {
  MinervaSystem::Instance().Initialize(argc, argv);
  int n = 10; // num samples
  int k = 8; // num features
  //NArray x = NArray::Randn({n, k}, 0.0, 1.0, {1, 2});
  //NArray y = NArray::Randn({n, k}, 0.0, 1.0, {1, 2});
  //NArray theta = NArray::Randn({k, k}, 0.0, 1.0, {2, 2});
  SimpleFileLoader loader;
  NArray x = NArray::LoadFromFile({n, k}, "x.dat", &loader, {1, 1});
  NArray y = NArray::LoadFromFile({n, k}, "y.dat", &loader, {1, 1});
  NArray theta = NArray::LoadFromFile({k, k}, "theta.dat", &loader, {1, 1});
  float alpha = 0.5; // lr
  int epoch = 2;
  for(int i = 0; i < epoch; ++i) {
    NArray error = x * theta - y;
    theta = theta - alpha * x.Trans() * error;
  }
  FileFormat format; format.binary = false;
  theta.ToFile("theta_trained.txt", format);
  format.binary = true;
  theta.ToFile("theta_trained.dat", format);
  return 0;
}
