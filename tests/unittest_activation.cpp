#include "unittest_main.h"
#include <cmath>

using namespace std;
using namespace minerva;

TEST(Activation, SigmoidForward) {
  auto& ms = MinervaSystem::Instance();
  Scale input_size{7, 6, 3, 2};

  ms.current_device_id_ = cpu_device;
  ImageBatch input = NArray::Randn(input_size, 0, 1);
  ms.current_device_id_ = gpu_device;
  ImageBatch output = Convolution::ActivationForward(input, ActivationAlgorithm::kSigmoid);
  auto input_ptr = input.Get();
  auto output_ptr = output.Get();
  for (int i = 0; i < input_size.Prod(); ++i) {
    EXPECT_FLOAT_EQ(output_ptr.get()[i], 1 / (1 + exp(-input_ptr.get()[i])));
  }
}

TEST(Activation, ReluForward) {
  auto& ms = MinervaSystem::Instance();
  Scale input_size{7, 6, 3, 2};

  ms.current_device_id_ = cpu_device;
  ImageBatch input = NArray::Randn(input_size, 0, 1);
  ms.current_device_id_ = gpu_device;
  ImageBatch output = Convolution::ActivationForward(input, ActivationAlgorithm::kRelu);
  auto input_ptr = input.Get();
  auto output_ptr = output.Get();
  for (int i = 0; i < input_size.Prod(); ++i) {
    EXPECT_FLOAT_EQ(output_ptr.get()[i], 0 < input_ptr.get()[i] ? input_ptr.get()[i] : 0);
  }
}

TEST(Activation, TanhForward) {
  auto& ms = MinervaSystem::Instance();
  Scale input_size{7, 6, 3, 2};

  ms.current_device_id_ = cpu_device;
  ImageBatch input = NArray::Randn(input_size, 0, 1);
  ms.current_device_id_ = gpu_device;
  ImageBatch output = Convolution::ActivationForward(input, ActivationAlgorithm::kTanh);
  auto input_ptr = input.Get();
  auto output_ptr = output.Get();
  for (int i = 0; i < input_size.Prod(); ++i) {
    EXPECT_FLOAT_EQ(output_ptr.get()[i], tanh(input_ptr.get()[i]));
  }
}
