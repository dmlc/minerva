#include <minerva.h>
#include <gtest/gtest.h>
#include <memory>

using namespace std;
using namespace minerva;

TEST(EvalTest, SyncEval) {
  NArray a = NArray::Randn({250, 500}, 0.0, 1.0);
  NArray b = NArray::Randn({500, 400}, 0.0, 1.0);
  NArray c = a * b;
  float* x = new float[250 * 500];
  float* y = new float[500 * 400];
  float* z = new float[250 * 400];
  for(int i = 0; i < 250; ++i) {
    for(int j = 0; j < 400; ++j) {
      z[i * 400 + j] = 0;
      for(int k = 0; k < 500; ++k) {
        z[i * 400 + j] += x[i * 500 + k] * y[k * 400 + j];
      }
    }
  }
  delete[] x;
  delete[] y;
  delete[] z;
}

TEST(EvalTest, AsyncEval) {
  NArray a = NArray::Randn({250, 500}, 0.0, 1.0);
  NArray b = NArray::Randn({500, 400}, 0.0, 1.0);
  NArray c = a * b;
  float* x = new float[250 * 500];
  float* y = new float[500 * 400];
  float* z = new float[250 * 400];
  for(int i = 0; i < 250; ++i) {
    for(int j = 0; j < 400; ++j) {
      z[i * 400 + j] = 0;
      for(int k = 0; k < 500; ++k) {
        z[i * 400 + j] += x[i * 500 + k] * y[k * 400 + j];
      }
    }
  }
  delete [] x;
  delete [] y;
  delete [] z;
  MinervaSystem::Instance().backend().WaitForAll();
}

TEST(EvalTest, AsyncEvalWithChangedDag) {
  NArray a = NArray::Zeros({250, 500});
  NArray b = NArray::Zeros({500, 400});
  NArray c = a * b;
  NArray d = c + 1;
  NArray e = b * d.Trans(); // 500x250
  MinervaSystem::Instance().backend().WaitForAll();
  shared_ptr<float> eptr = e.Get();
  for (int i = 0; i < 500 * 250; ++i) {
    ASSERT_EQ(eptr.get()[i], 0.0);
  }
}

TEST(WaitFinishTest, OnlyWait) {
  MinervaSystem::Instance().backend().WaitForAll();
}

