#include <iostream>
#include <minerva.h>

using namespace minerva;
using namespace std;

void Test1() {
  cout << "Max1" << endl;
  Scale size{5, 3};
  float* data = new float[size.Prod()];
  for (int i = 0; i < size.Prod(); ++i) {
    data[i] = i;
  }
  NArray na = NArray::LoadFromArray(size, data, {2, 2});
  FileFormat format;
  format.binary = false;
  na.ToStream(cout, format);
  std::cout << std::endl;
  NArray na2 = na.Max(1);
  na2.ToStream(cout, format);
  std::cout << std::endl;
  delete[] data;
}

void Test2() {
  cout << "Max2" << endl;
  Scale size{5, 3};
  float* data = new float[size.Prod()];
  for (int i = 0; i < size.Prod(); ++i) {
    data[i] = i;
  }
  NArray na = NArray::LoadFromArray(size, data, {2, 2});
  FileFormat format;
  format.binary = false;
  na.ToStream(cout, format);
  std::cout << std::endl;
  NArray na2 = na.Max(0);
  na2.ToStream(cout, format);
  std::cout << std::endl;
  delete[] data;
}

void Test3() {
  cout << "Sum1" << endl;
  Scale size{5, 3};
  float* data = new float[size.Prod()];
  for (int i = 0; i < size.Prod(); ++i) {
    data[i] = i;
  }
  NArray na = NArray::LoadFromArray(size, data, {2, 2});
  FileFormat format;
  format.binary = false;
  na.ToStream(cout, format);
  std::cout << std::endl;
  NArray na2 = na.Sum(1);
  na2.ToStream(cout, format);
  std::cout << std::endl;
  delete[] data;
}

void Test4() {
  cout << "Sum2" << endl;
  Scale size{5, 3};
  float* data = new float[size.Prod()];
  for (int i = 0; i < size.Prod(); ++i) {
    data[i] = i;
  }
  NArray na = NArray::LoadFromArray(size, data, {2, 2});
  FileFormat format;
  format.binary = false;
  na.ToStream(cout, format);
  std::cout << std::endl;
  NArray na2 = na.Sum(0);
  na2.ToStream(cout, format);
  std::cout << std::endl;
  delete[] data;
}
int main(int argc, char** argv) {
  MinervaSystem::Instance().Initialize(argc, argv);
  Test1();
  Test2();
  Test3();
  Test4();
  return 0;
}

