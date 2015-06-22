#include "unittest_main.h"

#ifdef HAS_CUDA
TEST(AllReduce, OnGpu) {
  auto&& ms = minerva::MinervaSystem::Instance();
  size_t single_device_num = 5;
  std::vector<std::vector<minerva::NArray>> inputs;
  minerva::Scale size{2, 2};
  std::vector<std::vector<float>> expect;
  for (size_t i = 0; i < single_device_num; ++i) {
    std::vector<float> v(size.Prod());
    expect.emplace_back(std::move(v));
  }
  for (size_t i = 0; i < gpu_devices.size(); ++i) {
    ms.SetDevice(gpu_devices.at(i));
    std::vector<minerva::NArray> input;
    for (size_t j = 0; j < single_device_num; ++j) {
      auto n = minerva::NArray::Randn(size, 0, 1);
      auto ptr = n.Get();
      for (int k = 0; k < size.Prod(); ++k) {
        expect.at(j).at(k) += ptr.get()[k];
      }
      input.emplace_back(std::move(n));
    }
    inputs.emplace_back(std::move(input));
  }
  auto func = [](minerva::NArray const& a, minerva::NArray const& b) {
    return a + b;
  };
  auto rst = minerva::algorithm::AllReduce(inputs, func);
  for (size_t i = 0; i < rst.size(); ++i) {
    auto ptr = rst.at(i).Get();
    for (int j = 0; j < rst.at(i).Size().Prod(); ++j) {
      EXPECT_FLOAT_EQ(ptr.get()[j], expect.at(i).at(j));
    }
  }
}

#endif
