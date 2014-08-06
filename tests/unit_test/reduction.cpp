#include <iostream>
#include <minerva.h>

using namespace minerva;
using namespace std;

void LoadAndOutput() {
  SimpleFileLoader loader;
  NArray na = NArray::LoadFromFile({4, 6}, "randmat.dat", &loader, {2, 2});

  FileFormat format;
  format.binary = false;
  na.ToFile("randmat1.txt", format);

  cout << "physical dag: " << endl;
  cout << MinervaSystem::Instance().physical_dag().PrintDag() << endl;
}

void Test1() {
  Scale size{5, 3};
  float* data = new float[size.Prod()];
  for (int i = 0; i < size.Prod(); ++i) {
    data[i] = i;
  }
  NArray na = NArray::LoadFromArray(size, data, {2, 2});
  FileFormat format;
  format.binary = false;
  na.ToStream(cout, format);
  delete[] data;
}

int main(int argc, char** argv) {
  MinervaSystem::Instance().Initialize(argc, argv);
  Test1();
  return 0;
}
