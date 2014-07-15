#include "common/nvector.h"
#include <iostream>

using namespace std;
using namespace minerva;

void Test1() {
  NVector<int> nv1({3, 4});
  for(int i = 0; i < 3; ++i)
    for(int j = 0; j < 4; ++j)
      nv1[{i, j}] = i + j;
  for(int i = 0; i < 3; ++i) {
    for(int j = 0; j < 4; ++j) {
      cout << nv1[{i, j}] << " ";
    }
    cout << endl;
  }
  NVector<int> nv2 = nv1.Map<int>(
      [](const int& v)->int { cout << v << endl; return v+1; }
  );
  for(int i = 0; i < 3; ++i) {
    for(int j = 0; j < 4; ++j) {
      cout << nv2[{i, j}] << " ";
    }
    cout << endl;
  }
}

void Test2() {
  Scale s1 = {4, 5};
  Scale s2 = {3, 3};
  NVector<Scale> s3 = s1.EquallySplit(s2);
  cout << s3.Size() << endl;
  for(int i = 0; i < s3.Size()[0]; ++i) {
    cout << "|| ";
    for(int j = 0; j < s3.Size()[1]; ++j) {
      cout << s3[{i, j}] << " || ";
    }
    cout << endl;
  }
}

int main() {
  cout << ">>>Test1..." << endl;
  Test1();
  cout << ">>>Test1 passed" << endl;
  cout << ">>>Test2..." << endl;
  Test2();
  cout << ">>>Test2 passed" << endl;
  return 0;
}
