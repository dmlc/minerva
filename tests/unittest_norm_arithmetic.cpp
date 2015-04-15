#include "unittest_main.h"

using namespace minerva;
using namespace std;

TEST(NormArithmetic, CpuAddOnFirstDimension) {
  auto& ms = MinervaSystem::Instance();
  ms.SetDevice(cpu_device);
  Scale s1{9, 7};
  Scale s2{1, 7};
  NArray n1 = NArray::Randn(s1, 0, 1);
  NArray n2 = NArray::Randn(s2, 0, 1);
  NArray n3 = n1.NormArithmetic(n2, ArithmeticType::kAdd);
  auto n1_ptr = n1.Get();
  auto n2_ptr = n2.Get();
  auto n3_ptr = n3.Get();
  for (int i = 0; i < n1.Size().Prod(); ++i) {
    EXPECT_FLOAT_EQ(n3_ptr.get()[i], n1_ptr.get()[i] + n2_ptr.get()[i / 9]);
  }
}

TEST(NormArithmetic, GpuMultSecondDimension) {
  auto& ms = MinervaSystem::Instance();
  ms.SetDevice(cpu_device);
  Scale s1{9, 7};
  Scale s2{9, 1};
  NArray n1 = NArray::Randn(s1, 0, 1);
  NArray n2 = NArray::Randn(s2, 0, 1);
  NArray n3 = n1.NormArithmetic(n2, ArithmeticType::kMult);
  auto n1_ptr = n1.Get();
  auto n2_ptr = n2.Get();
  auto n3_ptr = n3.Get();
  for (int i = 0; i < n1.Size().Prod(); ++i) {
    EXPECT_FLOAT_EQ(n3_ptr.get()[i], n1_ptr.get()[i] * n2_ptr.get()[i % 9]);
  }
}

