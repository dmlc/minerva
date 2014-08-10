#include <iostream>
#include <minerva.h>

using namespace minerva;
using namespace std;

void Test1() {
  Scale size{5, 3};
  float* data = new float[size.Prod()];
  for (int i = 0; i < size.Prod(); ++i) {
    data[i] = i;
  }
  NArray na = NArray::LoadFromArray(size, data, {2, 2});
  NArray na2 = NArray::Constant({5, 1}, 0.5, {2, 1});
  na.NormArithmetic(na2, ADD);
  cout << MinervaSystem::Instance().logical_dag().PrintDag() << endl;
  delete[] data;
}

int main(int argc, char** argv) {
  MinervaSystem::Instance().Initialize(&argc, &argv);
  Test1();
  return 0;
}

