#include <minerva.h>
#include <gtest/gtest.h>
#include <fstream>

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

class MiniBatchIOTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    ofstream fout(mb_file_name.c_str());
    fout.write((char*)&num_samples, 4);
    fout.write((char*)&sample_length, 4);
    for(int i = 0; i < num_samples; ++i) {
      float val = i;
      for(int j = 0; j < sample_length; ++j) {
        fout.write((char*)&val, 4);
      }
    }
  }
  static void TearDownTestCase() {
  }
  static const string mb_file_name;
  static const int num_samples, sample_length;
};
const string MiniBatchIOTest::mb_file_name = "test_mb.dat";
const int MiniBatchIOTest::num_samples = 100;
const int MiniBatchIOTest::sample_length = 10;

TEST_F(MiniBatchIOTest, NormalRead) {
  OneFileMBLoader loader(mb_file_name, {sample_length});
  EXPECT_EQ(loader.num_samples(), num_samples);

  int sample_idx = 0;
  NArray a1 = loader.LoadNext(10);
  float* a1ptr = a1.Get();
  for(int i = 0; i < 10; ++i,++sample_idx)
    for(int j = 0; j < sample_length; ++j)
      EXPECT_EQ(a1ptr[j + i * sample_length], sample_idx);
  delete [] a1ptr;

  NArray a2 = loader.LoadNext(10);
  float* a2ptr = a2.Get();
  for(int i = 0; i < 10; ++i,++sample_idx)
    for(int j = 0; j < sample_length; ++j)
      EXPECT_EQ(a2ptr[j + i * sample_length], sample_idx);
  delete [] a2ptr;
}

TEST_F(MiniBatchIOTest, ReadWrapAround) {
}
