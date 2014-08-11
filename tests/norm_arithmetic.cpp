#include <iostream>
#include <minerva.h>

using namespace minerva;
using namespace std;

void Test1() {
  cout << "NormArithmetic DAG test" << endl;
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

void Test2() {
  cout << "NormArithmetic test on first dimension" << endl;
  Scale s1{9, 7};
  Scale s2{1, 7};
  float* data = new float[s1.Prod()];
  for (int i = 0; i < s1.Prod(); ++i) {
    data[i] = i;
  }
  NArray n1 = NArray::LoadFromArray(s1, data, {2, 2});
  FileFormat format;
  format.binary = false;
  cout << "N1" << endl;
  n1.ToStream(cout, format);
  cout << endl;
  NArray n2 = NArray::Constant(s2, 2, {1, 2});
  cout << "N2" << endl;
  n2.ToStream(cout, format);
  cout << endl;
  NArray n3 = n1.NormArithmetic(n2, ADD);
  cout << "Result" << endl;
  n3.ToStream(cout, format);
  cout << endl;
  delete[] data;
}

void Test3() {
  cout << "NormArithmetic test on first dimension" << endl;
  Scale s1{9, 7};
  Scale s2{1, 7};
  float* data = new float[s1.Prod()];
  for (int i = 0; i < s1.Prod(); ++i) {
    data[i] = i;
  }
  NArray n1 = NArray::LoadFromArray(s1, data, {2, 2});
  FileFormat format;
  format.binary = false;
  cout << "N1" << endl;
  n1.ToStream(cout, format);
  cout << endl;
  float* data2 = new float[s2.Prod()];
  for (int i = 0; i < s2.Prod(); ++i) {
    data2[i] = 9 - i;
  }
  NArray n2 = NArray::LoadFromArray(s2, data2, {1, 2});
  cout << "N2" << endl;
  n2.ToStream(cout, format);
  cout << endl;
  NArray n3 = n1.NormArithmetic(n2, ADD);
  cout << "Result" << endl;
  n3.ToStream(cout, format);
  cout << endl;
  delete[] data2;
  delete[] data;
}

void Test4() {
  cout << "NormArithmetic test on second dimension" << endl;
  Scale s1{9, 7};
  Scale s2{9, 1};
  float* data = new float[s1.Prod()];
  for (int i = 0; i < s1.Prod(); ++i) {
    data[i] = i;
  }
  NArray n1 = NArray::LoadFromArray(s1, data, {2, 2});
  FileFormat format;
  format.binary = false;
  cout << "N1" << endl;
  n1.ToStream(cout, format);
  cout << endl;
  float* data2 = new float[s2.Prod()];
  for (int i = 0; i < s2.Prod(); ++i) {
    data2[i] = 9 - i;
  }
  NArray n2 = NArray::LoadFromArray(s2, data2, {2, 1});
  cout << "N2" << endl;
  n2.ToStream(cout, format);
  cout << endl;
  NArray n3 = n1.NormArithmetic(n2, MULT);
  cout << "Result" << endl;
  n3.ToStream(cout, format);
  cout << endl;
  delete[] data2;
  delete[] data;
}
int main(int argc, char** argv) {
  MinervaSystem::Instance().Initialize(&argc, &argv);
  Test1();
  Test2();
  Test3();
  Test4();
  return 0;
}

