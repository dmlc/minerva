#include "unittest_main.h"
#include <cmath>

using namespace std;
using namespace minerva;

TEST(Elewise, CpuExp) {
  auto& ms = MinervaSystem::Instance();
  ms.current_device_id_ = cpu_device;
  Scale size{2, 3, 4, 5, 6};

  auto a = NArray::Randn(size, 0, 5);
  auto b = Elewise::Exp(a);
  auto a_ptr = a.Get();
  auto b_ptr = b.Get();
  for (int i = 0; i < size.Prod(); ++i) {
    EXPECT_FLOAT_EQ(b_ptr.get()[i], exp(a_ptr.get()[i]));
  }
}

TEST(Elewise, GpuExp) {
  auto& ms = MinervaSystem::Instance();
  ms.current_device_id_ = cpu_device;
  Scale size{2, 3, 4, 5, 6};

  auto a = NArray::Randn(size, 0, 5);
  ms.current_device_id_ = gpu_device;
  auto b = Elewise::Exp(a);
  auto a_ptr = a.Get();
  auto b_ptr = b.Get();
  for (int i = 0; i < size.Prod(); ++i) {
    EXPECT_FLOAT_EQ(b_ptr.get()[i], exp(a_ptr.get()[i]));
  }
}

TEST(Elewise, CpuLn) {
  auto& ms = MinervaSystem::Instance();
  ms.current_device_id_ = cpu_device;
  Scale size{2, 3, 4, 5, 6};

  auto a = NArray::Randn(size, 500, 1);
  auto b = Elewise::Ln(a);
  auto a_ptr = a.Get();
  auto b_ptr = b.Get();
  for (int i = 0; i < size.Prod(); ++i) {
    EXPECT_FLOAT_EQ(b_ptr.get()[i], log(a_ptr.get()[i]));
  }
}

TEST(Elewise, GpuLn) {
  auto& ms = MinervaSystem::Instance();
  ms.current_device_id_ = cpu_device;
  Scale size{2, 3, 4, 5, 6};

  auto a = NArray::Randn(size, 500, 1);
  ms.current_device_id_ = gpu_device;
  auto b = Elewise::Ln(a);
  auto a_ptr = a.Get();
  auto b_ptr = b.Get();
  for (int i = 0; i < size.Prod(); ++i) {
    EXPECT_FLOAT_EQ(b_ptr.get()[i], log(a_ptr.get()[i]));
  }
}

TEST(Elewise, CpuSigmoid) {
  auto& ms = MinervaSystem::Instance();
  ms.current_device_id_ = cpu_device;
  Scale size{2, 3, 4, 5, 6};

  auto a = NArray::Randn(size, 0, 5);
  auto b = Elewise::Sigmoid(a);
  auto a_ptr = a.Get();
  auto b_ptr = b.Get();
  for (int i = 0; i < size.Prod(); ++i) {
    EXPECT_FLOAT_EQ(b_ptr.get()[i], 1 / (1 + exp(-a_ptr.get()[i])));
  }
}

TEST(Elewise, GpuSigmoid) {
  auto& ms = MinervaSystem::Instance();
  ms.current_device_id_ = cpu_device;
  Scale size{2, 3, 4, 5, 6};

  auto a = NArray::Randn(size, 0, 5);
  ms.current_device_id_ = gpu_device;
  auto b = Elewise::Sigmoid(a);
  auto a_ptr = a.Get();
  auto b_ptr = b.Get();
  for (int i = 0; i < size.Prod(); ++i) {
    EXPECT_FLOAT_EQ(b_ptr.get()[i], 1 / (1 + exp(-a_ptr.get()[i])));
  }
}

TEST(Elewise, CpuNegative) {
  auto& ms = MinervaSystem::Instance();
  ms.current_device_id_ = cpu_device;
  Scale size{2, 3, 4, 5, 6};

  auto a = NArray::Randn(size, 0, 5);
  auto b = -a;
  auto a_ptr = a.Get();
  auto b_ptr = b.Get();
  for (int i = 0; i < size.Prod(); ++i) {
    EXPECT_FLOAT_EQ(b_ptr.get()[i], -a_ptr.get()[i]);
  }
}

TEST(Elewise, GpuDiv) {
  auto& ms = MinervaSystem::Instance();
  ms.current_device_id_ = cpu_device;
  Scale size{2, 3, 4, 5, 6};

  auto a = NArray::Randn(size, 0, 5);
  ms.current_device_id_ = gpu_device;
  auto b = -a;
  auto a_ptr = a.Get();
  auto b_ptr = b.Get();
  for (int i = 0; i < size.Prod(); ++i) {
    EXPECT_FLOAT_EQ(b_ptr.get()[i], -a_ptr.get()[i]);
  }
}

