#include <iostream>
#include <minerva.h>
#include <gtest/gtest.h>

using namespace minerva;
using namespace std;

extern uint64_t cpuDevice;
extern uint64_t gpuDevice;

TEST(NormArithmeticGPU, AddFirstDimension1) {
  Scale s1{9, 7};
  Scale s2{1, 7};
  shared_ptr<float> data( new float[s1.Prod()] );
  for (int i = 0; i < s1.Prod(); ++i) {
    data.get()[i] = i;
  }
  NArray n1 = NArray::MakeNArray(s1, data, {1, 1});
  NArray n2 = NArray::Constant(s2, 2, {1, 1});
  MinervaSystem& ms = MinervaSystem::Instance();
  ms.set_device_id(gpuDevice);
  NArray n3 = n1.NormArithmetic(n2, ADD);
  float* res = n3.Get();
  for (int i = 0; i < n1.Size().Prod(); ++i) {
    EXPECT_FLOAT_EQ(res[i], i + 2);
  }
}

TEST(NormArithmeticGPU, AddFirstDimension2) {
  Scale s1{9, 7};
  Scale s2{1, 7};
  shared_ptr<float> data( new float[s1.Prod()] );
  for (int i = 0; i < s1.Prod(); ++i) {
    data.get()[i] = i;
  }
  NArray n1 = NArray::MakeNArray(s1, data, {1, 1});
  shared_ptr<float> data2( new float[s2.Prod()] );
  for (int i = 0; i < s2.Prod(); ++i) {
    data2.get()[i] = 9 - i;
  }
  NArray n2 = NArray::MakeNArray(s2, data2, {1, 1});
  MinervaSystem& ms = MinervaSystem::Instance();
  ms.set_device_id(gpuDevice);
  NArray n3 = n1.NormArithmetic(n2, ADD);
  float* res = n3.Get();
  for (int i = 0; i < n1.Size().Prod(); ++i) {
    EXPECT_FLOAT_EQ(res[i], 9 - (i / 9) + i);
  }
}

TEST(NormArithmeticGPU, MultSecondDimension) {
  Scale s1{9, 7};
  Scale s2{9, 1};
  shared_ptr<float> data(new float[s1.Prod()]);
  for (int i = 0; i < s1.Prod(); ++i) {
    data.get()[i] = i;
  }
  NArray n1 = NArray::MakeNArray(s1, data, {1, 1});
  shared_ptr<float> data2(new float[s2.Prod()]);
  for (int i = 0; i < s2.Prod(); ++i) {
    data2.get()[i] = 9 - i;
  }
  NArray n2 = NArray::MakeNArray(s2, data2, {1, 1});
  MinervaSystem& ms = MinervaSystem::Instance();
  ms.set_device_id(gpuDevice);
  NArray n3 = n1.NormArithmetic(n2, MULT);
  float* res = n3.Get();
  for (int i = 0; i < 9; ++i) {
    for (int j = 0; j < 7; ++j) {
      EXPECT_FLOAT_EQ(res[i + 9 * j], (i + 9 * j) * (9 - i));
    }
  }
}

