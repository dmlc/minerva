#include <iostream>
#include <iomanip>
#include <minerva.h>
#include <op/impl/basic.h>
#include <gtest/gtest.h>

using namespace minerva;
using namespace std;

void FillIncr(float* ptr, size_t len) {
  for(size_t i = 0; i < len; ++i) ptr[i] = i+1;
}

void Fill(float* ptr, size_t len, float val) {
  for(size_t i = 0; i < len; ++i) ptr[i] = val;
}

/*void CallNCopy(
    float* src, const Scale& srcsize, const Scale& srcstart,
    float* dst, const Scale& dstsize, const Scale& dststart,
    const Scale& copysize) {
  cout << "call NCopy:\n  srcsize=" << srcsize << " srcstart=" << srcstart
    << " dstsize=" << dstsize << " dststart=" << dststart
    << " copysize=" << copysize << endl;
  basic::NCopy(src, srcsize, srcstart, dst, dstsize, dststart, copysize);
}*/

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

struct Case {
  Scale srcsize, srcst, dstsize, dstst, copysize;
  const float* ans;
};

std::ostream& operator << (std::ostream&os, const Case& c) {
  return os << "{ srcsize=" << c.srcsize << " srcstart=" << c.srcst
    << " dstsize=" << c.dstsize << " dststart=" << c.dstst
    << " copysize=" << c.copysize << " }";
}

/* src array = {
   1,  2,  3,  4,
   5,  6,  7,  8,
   9, 10, 11, 12,
  13, 14, 15, 16,
  17, 18, 19, 20,
 }
*/

const float ans1[] = { 
   1,  2,  3,  4,  0,
   5,  6,  7,  8,  0,
   9, 10, 11, 12,  0,
  13, 14, 15, 16,  0,
  17, 18, 19, 20,  0,
   0,  0,  0,  0,  0
};

const float ans2[] = { 
   1,  2,  3,  4,
   5,  6,  7,  8,
   9, 10, 11, 12,
  13, 14, 15, 16,
  17, 18, 19, 20,
   0,  0,  0,  0,
};

const float ans3[] = { 
   0,  0,  0,  0,
   1,  2,  3,  4,
   5,  6,  7,  8,
   9, 10, 11, 12,
  13, 14, 15, 16,
  17, 18, 19, 20,
};

const float ans4[] = { 
   1,  2,  0,  0,  0,
   5,  6,  0,  0,  0,
   9, 10,  0,  0,  0,
   0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,
   0,  0,  0,  0,  0
};

const float ans5[] = { 
  11, 12,  0,  0,  0,
  15, 16,  0,  0,  0,
  19, 20,  0,  0,  0,
   0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,
};

const float ans6[] = { 
   0,  0,  0,  0,  0,
   0,  0,  0, 11, 12,
   0,  0,  0, 15, 16,
   0,  0,  0, 19, 20,
   0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,
};

const Case inputs[] = {
  {{4, 5}, {0, 0}, {5, 6}, {0, 0}, {4, 5}, ans1},
  {{4, 5}, {0, 0}, {4, 6}, {0, 0}, {4, 5}, ans2},
  {{4, 5}, {0, 0}, {4, 6}, {0, 1}, {4, 5}, ans3},
  {{4, 5}, {0, 0}, {5, 6}, {0, 0}, {2, 3}, ans4},
  {{4, 5}, {2, 2}, {5, 6}, {0, 0}, {2, 3}, ans5},
  {{4, 5}, {2, 2}, {5, 6}, {3, 1}, {2, 3}, ans6},
};

class NCopyTest : public testing::TestWithParam<Case> { };

TEST_P(NCopyTest, Test2DCopy) {
  const Case& c = GetParam();
  //Scale srcsize, Scale srcst, Scale dstsize, Scale dstst, Scale copysize) {
  float* src = new float[c.srcsize.Prod()];
  float* dst = new float[c.dstsize.Prod()];
  FillIncr(src, c.srcsize.Prod());
  Fill(dst, c.dstsize.Prod(), 0.0);
  basic::NCopy(src, c.srcsize, c.srcst, dst, c.dstsize, c.dstst, c.copysize);
  for(int i = 0; i < c.dstsize.Prod(); ++i) {
    EXPECT_EQ(dst[i], c.ans[i]) << "wrong answer at i=" << i;
  }
  // print
  //cout << "src=" << endl;
  //Print2D(src, srcsize);
  //cout << "dst=" << endl;
  //Print2D(dst, dstsize);
}

INSTANTIATE_TEST_CASE_P(2DCopy, NCopyTest, ::testing::ValuesIn(inputs));
