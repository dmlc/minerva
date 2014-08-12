#include <minerva.h>
#include <gtest/gtest.h>

using namespace minerva;
using namespace std;

TEST(NArrayFileIO, Correctness) {
  NArray na = NArray::Randn({4, 6}, 0.0, 1.0, {2, 2});
  float* val = na.Get();
  // save
  FileFormat format;
  format.binary = true;
  na.ToFile("randmat.dat", format);
  // load
  SimpleFileLoader loader;
  NArray na1 = NArray::LoadFromFile({4, 6}, "randmat.dat", &loader, {2, 2});
  float* val1 = na1.Get();
  // check
  for(int i = 0; i < 4*6; ++i) {
    EXPECT_FLOAT_EQ(val[i], val1[i]) << "different val at i=" << i;
  }
}

TEST(NArrayMiniBatchIO, Correctness) {
}
