#include <iostream>
#include <iomanip>
#include <minerva.h>
#include <op/impl/basic.h>

using namespace minerva;
using namespace std;

void FillIncr(float* ptr, size_t len) {
  for(size_t i = 0; i < len; ++i) ptr[i] = i+1;
}

void Fill(float* ptr, size_t len, float val) {
  for(size_t i = 0; i < len; ++i) ptr[i] = val;
}

void CallNCopy(
    float* src, const Scale& srcsize, const Scale& srcstart,
    float* dst, const Scale& dstsize, const Scale& dststart,
    const Scale& copysize) {
  cout << "call NCopy:\n  srcsize=" << srcsize << "srcstart=" << srcstart
    << "\n  dstsize=" << dstsize << "dststart=" << dststart
    << "\n  copysize=" << copysize << endl;
  basic::NCopy(src, srcsize, srcstart, dst, dstsize, dststart, copysize);
}

void Print2D(float* ptr, Scale size) {
  for(int i = 0; i < size[0]; ++i) {
    cout << "  | ";
    for(int j = 0; j < size[1]; ++j) {
      cout << std::setw(2) << setprecision(2) << ptr[i + j * size[0]] << " ";
    }
    cout << "|" << endl;
  }
  cout << endl;
}

void Test1(Scale srcsize, Scale srcst, Scale dstsize, Scale dstst, Scale copysize) {
  cout << "Test 2D NCopy" << endl;
  float* src = new float[srcsize.Prod()];
  float* dst = new float[dstsize.Prod()];
  FillIncr(src, srcsize.Prod());
  Fill(dst, dstsize.Prod(), 0.0);
  CallNCopy(src, srcsize, srcst, dst, dstsize, dstst, copysize);
  // print
  cout << "src=" << endl;
  Print2D(src, srcsize);
  cout << "dst=" << endl;
  Print2D(dst, dstsize);
}

int main(int argc, char** argv) {
  MinervaSystem::Instance().Initialize(&argc, &argv);
  Test1({4, 5}, {0, 0}, {5, 6}, {0, 0}, {4, 5});
  Test1({4, 5}, {0, 0}, {4, 6}, {0, 0}, {4, 5});
  Test1({4, 5}, {0, 0}, {4, 6}, {0, 1}, {4, 5});
  Test1({4, 5}, {0, 0}, {5, 6}, {0, 0}, {2, 3});
  Test1({4, 5}, {2, 2}, {5, 6}, {0, 0}, {2, 3});
  Test1({4, 5}, {2, 2}, {5, 6}, {3, 1}, {2, 3});
  return 0;
}
