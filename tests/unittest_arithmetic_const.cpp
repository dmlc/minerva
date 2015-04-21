#include "unittest_main.h"
#include <cmath>

using namespace std;
using namespace minerva;

TEST(ArithmeticConst, CpuAdd) {
  auto& ms = MinervaSystem::Instance();
  ms.SetDevice(cpu_device);
  Scale size{2, 3, 4, 5, 6};

  auto a = NArray::Randn(size, 0, 5);
  auto b = a + 32.9;
  auto a_ptr = a.Get();
  auto b_ptr = b.Get();
  for (int i = 0; i < size.Prod(); ++i) {
    EXPECT_FLOAT_EQ(b_ptr.get()[i], a_ptr.get()[i] + 32.9);
  }
}

#ifdef HAS_CUDA
TEST(ArithmeticConst, GpuAdd) {
  auto& ms = MinervaSystem::Instance();
  ms.SetDevice(cpu_device);
  Scale size{2, 3, 4, 5, 6};

  auto a = NArray::Randn(size, 0, 5);
  ms.SetDevice(gpu_device);
  auto b = 32.9 + a;
  auto a_ptr = a.Get();
  auto b_ptr = b.Get();
  for (int i = 0; i < size.Prod(); ++i) {
    EXPECT_FLOAT_EQ(b_ptr.get()[i], a_ptr.get()[i] + 32.9);
  }
}
#endif

TEST(ArithmeticConst, CpuSub) {
  auto& ms = MinervaSystem::Instance();
  ms.SetDevice(cpu_device);
  Scale size{2, 3, 4, 5, 6};

  auto a = NArray::Randn(size, 0, 5);
  auto b = a - 3.99;
  auto a_ptr = a.Get();
  auto b_ptr = b.Get();
  for (int i = 0; i < size.Prod(); ++i) {
    EXPECT_NEAR(b_ptr.get()[i], a_ptr.get()[i] - 3.99, 0.001);
  }
}

#ifdef HAS_CUDA
TEST(ArithmeticConst, GpuSub) {
  auto& ms = MinervaSystem::Instance();
  ms.SetDevice(cpu_device);
  Scale size{2, 3, 4, 5, 6};

  auto a = NArray::Randn(size, 0, 5);
  ms.SetDevice(gpu_device);
  auto b = 3.99 - a;
  auto a_ptr = a.Get();
  auto b_ptr = b.Get();
  for (int i = 0; i < size.Prod(); ++i) {
    EXPECT_NEAR(b_ptr.get()[i], 3.99 - a_ptr.get()[i], 0.001);
  }
}
#endif

TEST(ArithmeticConst, CpuMult) {
  auto& ms = MinervaSystem::Instance();
  ms.SetDevice(cpu_device);
  Scale size{2, 3, 4, 5, 6};

  auto a = NArray::Randn(size, 0, 5);
  auto b = a * 3.291;
  auto a_ptr = a.Get();
  auto b_ptr = b.Get();
  for (int i = 0; i < size.Prod(); ++i) {
    EXPECT_FLOAT_EQ(b_ptr.get()[i], a_ptr.get()[i] * 3.291);
  }
}

#ifdef HAS_CUDA
TEST(ArithmeticConst, GpuMult) {
  auto& ms = MinervaSystem::Instance();
  ms.SetDevice(cpu_device);
  Scale size{2, 3, 4, 5, 6};

  auto a = NArray::Randn(size, 0, 5);
  ms.SetDevice(gpu_device);
  auto b = 3.199 * a;
  auto a_ptr = a.Get();
  auto b_ptr = b.Get();
  for (int i = 0; i < size.Prod(); ++i) {
    EXPECT_FLOAT_EQ(b_ptr.get()[i], 3.199 * a_ptr.get()[i]);
  }
}
#endif

TEST(ArithmeticConst, CpuDiv) {
  auto& ms = MinervaSystem::Instance();
  ms.SetDevice(cpu_device);
  Scale size{2, 3, 4, 5, 6};

  auto a = NArray::Randn(size, 0, 5);
  auto b = a / 9.112;
  auto a_ptr = a.Get();
  auto b_ptr = b.Get();
  for (int i = 0; i < size.Prod(); ++i) {
    EXPECT_FLOAT_EQ(b_ptr.get()[i], a_ptr.get()[i] / 9.112);
  }
}

#ifdef HAS_CUDA
TEST(ArithmeticConst, GpuDiv) {
  auto& ms = MinervaSystem::Instance();
  ms.SetDevice(cpu_device);
  Scale size{2, 3, 4, 5, 6};

  auto a = NArray::Randn(size, 0, 5);
  ms.SetDevice(gpu_device);
  auto b = 3.331 / a;
  auto a_ptr = a.Get();
  auto b_ptr = b.Get();
  for (int i = 0; i < size.Prod(); ++i) {
    EXPECT_FLOAT_EQ(b_ptr.get()[i], 3.331 / a_ptr.get()[i]);
  }
}
#endif
