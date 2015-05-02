#include "unittest_main.h"
#include <cmath>

using namespace std;
using namespace minerva;

TEST(Elewise, CpuExp) {
  auto& ms = MinervaSystem::Instance();
  ms.SetDevice(cpu_device);
  Scale size{2, 3, 4, 5, 6};

  auto a = NArray::Randn(size, 0, 5);
  auto b = Elewise::Exp(a);
  auto a_ptr = a.Get();
  auto b_ptr = b.Get();
  for (int i = 0; i < size.Prod(); ++i) {
    EXPECT_FLOAT_EQ(b_ptr.get()[i], exp(a_ptr.get()[i]));
  }
}

#ifdef HAS_CUDA
TEST(Elewise, GpuExp) {
  auto& ms = MinervaSystem::Instance();
  ms.SetDevice(cpu_device);
  Scale size{2, 3, 4, 5, 6};

  auto a = NArray::Randn(size, 0, 5);
  ms.SetDevice(gpu_device);
  auto b = Elewise::Exp(a);
  auto a_ptr = a.Get();
  auto b_ptr = b.Get();
  for (int i = 0; i < size.Prod(); ++i) {
    EXPECT_FLOAT_EQ(b_ptr.get()[i], exp(a_ptr.get()[i]));
  }
}
#endif

TEST(Elewise, CpuLn) {
  auto& ms = MinervaSystem::Instance();
  ms.SetDevice(cpu_device);
  Scale size{2, 3, 4, 5, 6};

  auto a = NArray::Randn(size, 500, 1);
  auto b = Elewise::Ln(a);
  auto a_ptr = a.Get();
  auto b_ptr = b.Get();
  for (int i = 0; i < size.Prod(); ++i) {
    EXPECT_FLOAT_EQ(b_ptr.get()[i], log(a_ptr.get()[i]));
  }
}

#ifdef HAS_CUDA
TEST(Elewise, GpuLn) {
  auto& ms = MinervaSystem::Instance();
  ms.SetDevice(cpu_device);
  Scale size{2, 3, 4, 5, 6};

  auto a = NArray::Randn(size, 500, 1);
  ms.SetDevice(gpu_device);
  auto b = Elewise::Ln(a);
  auto a_ptr = a.Get();
  auto b_ptr = b.Get();
  for (int i = 0; i < size.Prod(); ++i) {
    EXPECT_FLOAT_EQ(b_ptr.get()[i], log(a_ptr.get()[i]));
  }
}

TEST(Elewise, GpuSigmoidForward) {
  MinervaSystem::Instance().SetDevice(gpu_device);
  Scale size{2, 3, 4, 5, 6};
  auto a = NArray::Randn(size, 0, 1);
  auto b = Elewise::SigmoidForward(a);
  auto c = Convolution::ActivationForward(a.Reshape({size.Prod(), 1, 1, 1}), ActivationAlgorithm::kSigmoid);
  auto a_ptr = a.Get();
  auto b_ptr = b.Get();
  auto c_ptr = c.Get();
  for (int i = 0; i < size.Prod(); ++i) {
    EXPECT_FLOAT_EQ(b_ptr.get()[i], 1 / (1 + exp(-a_ptr.get()[i])));
    EXPECT_FLOAT_EQ(b_ptr.get()[i], c_ptr.get()[i]);
  }
}

TEST(Elewise, GpuSigmoidBackward) {
  MinervaSystem::Instance().SetDevice(gpu_device);
  Scale size{2, 3, 4, 5, 6};
  auto top_diff = NArray::Randn(size, 0, 1);
  auto top = NArray::Randn(size, 0, 1);
  auto bottom = NArray::Randn(size, 0, 1);
  auto a = Elewise::SigmoidBackward(top_diff, top, bottom);
  auto b = Convolution::ActivationBackward(top_diff.Reshape({size.Prod(), 1, 1, 1}), top.Reshape({size.Prod(), 1, 1, 1}), bottom.Reshape({size.Prod(), 1, 1, 1}), ActivationAlgorithm::kSigmoid);
  auto a_ptr = a.Get();
  auto b_ptr = b.Get();
  for (int i = 0; i < size.Prod(); ++i) {
    EXPECT_FLOAT_EQ(b_ptr.get()[i], a_ptr.get()[i]);
  }
}

TEST(Elewise, GpuReluForward) {
  MinervaSystem::Instance().SetDevice(gpu_device);
  Scale size{2, 3, 4, 5, 6};
  auto a = NArray::Randn(size, 0, 1);
  auto b = Elewise::ReluForward(a);
  auto c = Convolution::ActivationForward(a.Reshape({size.Prod(), 1, 1, 1}), ActivationAlgorithm::kRelu);
  auto a_ptr = a.Get();
  auto b_ptr = b.Get();
  auto c_ptr = c.Get();
  for (int i = 0; i < size.Prod(); ++i) {
    EXPECT_FLOAT_EQ(b_ptr.get()[i], 0 < a_ptr.get()[i] ? a_ptr.get()[i] : 0);
    EXPECT_FLOAT_EQ(b_ptr.get()[i], c_ptr.get()[i]);
  }
}

TEST(Elewise, GpuReluBackward) {
  MinervaSystem::Instance().SetDevice(gpu_device);
  Scale size{2, 3, 4, 5, 6};
  auto top_diff = NArray::Randn(size, 0, 1);
  auto top = NArray::Randn(size, 0, 1);
  auto bottom = NArray::Randn(size, 0, 1);
  auto a = Elewise::ReluBackward(top_diff, top, bottom);
  auto b = Convolution::ActivationBackward(top_diff.Reshape({size.Prod(), 1, 1, 1}), top.Reshape({size.Prod(), 1, 1, 1}), bottom.Reshape({size.Prod(), 1, 1, 1}), ActivationAlgorithm::kRelu);
  auto a_ptr = a.Get();
  auto b_ptr = b.Get();
  for (int i = 0; i < size.Prod(); ++i) {
    EXPECT_FLOAT_EQ(b_ptr.get()[i], a_ptr.get()[i]);
  }
}

TEST(Elewise, GpuTanhForward) {
  MinervaSystem::Instance().SetDevice(gpu_device);
  Scale size{2, 3, 4, 5, 6};
  auto a = NArray::Randn(size, 0, 1);
  auto b = Elewise::TanhForward(a);
  auto c = Convolution::ActivationForward(a.Reshape({size.Prod(), 1, 1, 1}), ActivationAlgorithm::kTanh);
  auto a_ptr = a.Get();
  auto b_ptr = b.Get();
  auto c_ptr = c.Get();
  for (int i = 0; i < size.Prod(); ++i) {
    EXPECT_FLOAT_EQ(b_ptr.get()[i], tanh(a_ptr.get()[i]));
    EXPECT_FLOAT_EQ(b_ptr.get()[i], c_ptr.get()[i]);
  }
}

TEST(Elewise, GpuTanhBackward) {
  MinervaSystem::Instance().SetDevice(gpu_device);
  Scale size{2, 3, 4, 5, 6};
  auto top_diff = NArray::Randn(size, 0, 1);
  auto top = NArray::Randn(size, 0, 1);
  auto bottom = NArray::Randn(size, 0, 1);
  auto a = Elewise::TanhBackward(top_diff, top, bottom);
  auto b = Convolution::ActivationBackward(top_diff.Reshape({size.Prod(), 1, 1, 1}), top.Reshape({size.Prod(), 1, 1, 1}), bottom.Reshape({size.Prod(), 1, 1, 1}), ActivationAlgorithm::kTanh);
  auto a_ptr = a.Get();
  auto b_ptr = b.Get();
  for (int i = 0; i < size.Prod(); ++i) {
    EXPECT_FLOAT_EQ(b_ptr.get()[i], a_ptr.get()[i]);
  }
}
#endif

TEST(Elewise, CpuNegative) {
  auto& ms = MinervaSystem::Instance();
  ms.SetDevice(cpu_device);
  Scale size{2, 3, 4, 5, 6};

  auto a = NArray::Randn(size, 0, 5);
  auto b = -a;
  auto a_ptr = a.Get();
  auto b_ptr = b.Get();
  for (int i = 0; i < size.Prod(); ++i) {
    EXPECT_FLOAT_EQ(b_ptr.get()[i], -a_ptr.get()[i]);
  }
}

#ifdef HAS_CUDA
TEST(Elewise, GpuDiv) {
  auto& ms = MinervaSystem::Instance();
  ms.SetDevice(cpu_device);
  Scale size{2, 3, 4, 5, 6};

  auto a = NArray::Randn(size, 0, 5);
  ms.SetDevice(gpu_device);
  auto b = -a;
  auto a_ptr = a.Get();
  auto b_ptr = b.Get();
  for (int i = 0; i < size.Prod(); ++i) {
    EXPECT_FLOAT_EQ(b_ptr.get()[i], -a_ptr.get()[i]);
  }
}
#endif
