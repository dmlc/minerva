#include "unittest_main.h"
#include <cmath>

using namespace std;
using namespace minerva;

#ifdef HAS_CUDA
TEST(Activation, GpuSigmoidForward) {
  auto& ms = MinervaSystem::Instance();
  Scale input_size{7, 6, 3, 2};

  ms.SetDevice(cpu_device);
  ImageBatch input = NArray::Randn(input_size, 0, 1);
  ms.SetDevice(gpu_device);
  ImageBatch output = Convolution::ActivationForward(input, ActivationAlgorithm::kSigmoid);
  auto input_ptr = input.Get();
  auto output_ptr = output.Get();
  for (int i = 0; i < input_size.Prod(); ++i) {
    EXPECT_FLOAT_EQ(output_ptr.get()[i], 1 / (1 + exp(-input_ptr.get()[i])));
  }
}

TEST(Activation, GpuReluForward) {
  auto& ms = MinervaSystem::Instance();
  Scale input_size{7, 6, 3, 2};

  ms.SetDevice(cpu_device);
  ImageBatch input = NArray::Randn(input_size, 0, 1);
  ms.SetDevice(gpu_device);
  ImageBatch output = Convolution::ActivationForward(input, ActivationAlgorithm::kRelu);
  auto input_ptr = input.Get();
  auto output_ptr = output.Get();
  for (int i = 0; i < input_size.Prod(); ++i) {
    EXPECT_FLOAT_EQ(output_ptr.get()[i], 0 < input_ptr.get()[i] ? input_ptr.get()[i] : 0);
  }
}

TEST(Activation, GpuTanhForward) {
  auto& ms = MinervaSystem::Instance();
  Scale input_size{7, 6, 3, 2};

  ms.SetDevice(cpu_device);
  ImageBatch input = NArray::Randn(input_size, 0, 1);
  ms.SetDevice(gpu_device);
  ImageBatch output = Convolution::ActivationForward(input, ActivationAlgorithm::kTanh);
  auto input_ptr = input.Get();
  auto output_ptr = output.Get();
  for (int i = 0; i < input_size.Prod(); ++i) {
    EXPECT_FLOAT_EQ(output_ptr.get()[i], tanh(input_ptr.get()[i]));
  }
}
#endif

TEST(Activation, CpuSigmoidForward) {
  auto& ms = MinervaSystem::Instance();
  Scale input_size{7, 6, 3, 2};

  ms.SetDevice(cpu_device);
  ImageBatch input = NArray::Randn(input_size, 0, 1);
  ImageBatch output = Convolution::ActivationForward(input, ActivationAlgorithm::kSigmoid);
  auto input_ptr = input.Get();
  auto output_ptr = output.Get();
  for (int i = 0; i < input_size.Prod(); ++i) {
    EXPECT_FLOAT_EQ(output_ptr.get()[i], 1 / (1 + exp(-input_ptr.get()[i])));
  }
}

TEST(Activation, CpuReluForward) {
  auto& ms = MinervaSystem::Instance();
  Scale input_size{7, 6, 3, 2};

  ms.SetDevice(cpu_device);
  ImageBatch input = NArray::Randn(input_size, 0, 1);
  ImageBatch output = Convolution::ActivationForward(input, ActivationAlgorithm::kRelu);
  auto input_ptr = input.Get();
  auto output_ptr = output.Get();
  for (int i = 0; i < input_size.Prod(); ++i) {
    EXPECT_FLOAT_EQ(output_ptr.get()[i], 0 < input_ptr.get()[i] ? input_ptr.get()[i] : 0);
  }
}

TEST(Activation, CpuTanhForward) {
  auto& ms = MinervaSystem::Instance();
  Scale input_size{7, 6, 3, 2};

  ms.SetDevice(cpu_device);
  ImageBatch input = NArray::Randn(input_size, 0, 1);
  ImageBatch output = Convolution::ActivationForward(input, ActivationAlgorithm::kTanh);
  auto input_ptr = input.Get();
  auto output_ptr = output.Get();
  for (int i = 0; i < input_size.Prod(); ++i) {
    EXPECT_FLOAT_EQ(output_ptr.get()[i], tanh(input_ptr.get()[i]));
  }
}
