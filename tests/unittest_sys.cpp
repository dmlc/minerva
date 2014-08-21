#include <minerva.h>
#include <gtest/gtest.h>

using namespace std;
using namespace minerva;

TEST(EvalTest, SyncEval) {
  NArray a = NArray::Randn({250, 500}, 0.0, 1.0, {1, 1});
  NArray b = NArray::Randn({500, 400}, 0.0, 1.0, {1, 1});
  NArray c = a * b;
  cout << "Call sync eval" << endl;
  c.Eval();
  cout << "Call eval end" << endl;
  float* x = new float[250*500];
  float* y = new float[500*400];
  float* z = new float[250*400];
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
}

TEST(EvalTest, AsyncEval) {
  NArray a = NArray::Randn({250, 500}, 0.0, 1.0, {1, 1});
  NArray b = NArray::Randn({500, 400}, 0.0, 1.0, {1, 1});
  NArray c = a * b;
  cout << "Call async eval" << endl;
  c.EvalAsync();
  cout << "Call eval end" << endl;
  float* x = new float[250*500];
  float* y = new float[500*400];
  float* z = new float[250*400];
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
  MinervaSystem::Instance().WaitForEvalFinish();
}

TEST(EvalTest, AsyncEvalWithChangedLDag) {
  NArray a = NArray::Zeros({250, 500}, {2, 2});
  NArray b = NArray::Zeros({500, 400}, {2, 2});
  NArray c = a * b; // 250x400
  cout << "Call async eval" << endl;
  c.EvalAsync();
  cout << "Call eval end" << endl;
  NArray d = c + 1; // 250x400
  NArray e = b * d.Trans(); // 500x250
  MinervaSystem::Instance().WaitForEvalFinish();
  cout << "Call sync eval" << endl;
  float* eptr = e.Get();
  for (int i = 0; i < 500 * 250; ++i) {
    ASSERT_EQ(eptr[i], 0.0);
  }
  delete [] eptr;
  cout << "Call eval end" << endl;
}
