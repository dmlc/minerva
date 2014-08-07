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
  FileFormat format;
  format.binary = false;
  NArray na2 = na.Max(1);
  na2.ToStream(cout, format);
  delete[] data;
}

int main(int argc, char** argv) {
  MinervaSystem::Instance().Initialize(argc, argv);
  Test1();
  return 0;
}

