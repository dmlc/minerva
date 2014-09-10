#include <minerva.h>
#include <gtest/gtest.h>
#include <fstream>
#include <iostream>

using namespace minerva;
using namespace std;

TEST(NArrayFileIO, Correctness) {
  NArray na = NArray::Randn({4, 6}, 0.0, 1.0, {1, 1});
  float* val = na.Get();
  // save
  FileFormat format;
  format.binary = true;
  na.ToFile("randmat.dat", format);
  // load
  SimpleFileLoader loader;
  NArray na1 = NArray::LoadFromFile({4, 6}, "randmat.dat", &loader, {1, 1});
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
  void SetUp() {
  }
  void TearDown() {
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
  for(int k = 0; k < num_samples / 10; ++k) {
    NArray a1 = loader.LoadNext(10);
    float* a1ptr = a1.Get();
    for(int i = 0; i < 10; ++i,++sample_idx)
      for(int j = 0; j < sample_length; ++j)
        ASSERT_EQ(a1ptr[j + i * sample_length], sample_idx);
    delete [] a1ptr;
  }
}

TEST_F(MiniBatchIOTest, ReadWrapAround) {
  OneFileMBLoader loader(mb_file_name, {sample_length});
  EXPECT_EQ(loader.num_samples(), num_samples);
  int sample_idx = 0;
  for(int k = 0; k < num_samples / 10 * 2; ++k) {
    NArray a1 = loader.LoadNext(10);
    float* a1ptr = a1.Get();
    for(int i = 0; i < 10; ++i) {
      for(int j = 0; j < sample_length; ++j)
        ASSERT_EQ(a1ptr[j + i * sample_length], sample_idx);
      sample_idx = (sample_idx + 1) % num_samples;
    }
    delete [] a1ptr;
  }
}

TEST_F(MiniBatchIOTest, ReadWrapAroundCrossEdge) {
  OneFileMBLoader loader(mb_file_name, {sample_length});
  EXPECT_EQ(loader.num_samples(), num_samples);
  int sample_idx = 0;
  for(int k = 0; k < 10; ++k) {
    NArray a1 = loader.LoadNext(30);
    float* a1ptr = a1.Get();
    for(int i = 0; i < 30; ++i) {
      for(int j = 0; j < sample_length; ++j)
        ASSERT_EQ(a1ptr[j + i * sample_length], sample_idx);
      sample_idx = (sample_idx + 1) % num_samples;
    }
    delete [] a1ptr;
  }
}

TEST_F(MiniBatchIOTest, ReadWrapAroundMultipleAround) {
  OneFileMBLoader loader(mb_file_name, {sample_length});
  EXPECT_EQ(loader.num_samples(), num_samples);
  int sample_idx = 0;
  NArray a1 = loader.LoadNext(450);
  float* a1ptr = a1.Get();
  for(int i = 0; i < 450; ++i) {
    for(int j = 0; j < sample_length; ++j)
      ASSERT_EQ(a1ptr[j + i * sample_length], sample_idx);
    sample_idx = (sample_idx + 1) % num_samples;
  }
  delete [] a1ptr;
}

TEST_F(MiniBatchIOTest, LoadAndMult) {
  OneFileMBLoader loader(mb_file_name, {sample_length});
  EXPECT_EQ(loader.num_samples(), num_samples);
  NArray w = NArray::Constant({20, sample_length}, 0, {1, 1});
  NArray x = loader.LoadNext(30);
  NArray y = w * x;
  //cout << MinervaSystem::Instance().logical_dag().PrintDag() << endl;
  float* yptr = y.Get();
  for(int i = 0; i < 20 * 30; ++i)
    ASSERT_EQ(yptr[i], 0);
  delete [] yptr;
}
