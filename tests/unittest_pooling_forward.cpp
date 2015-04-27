#include "unittest_main.h"

using namespace std;
using namespace minerva;

#ifdef HAS_CUDA
TEST(PoolingForward, GpuWithoutPadding) {
  float input_raw[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  float correct_raw[] = {11, 12, 15, 16};
  auto& ms = MinervaSystem::Instance();
  Scale input_size{4, 4, 1, 1};
  Scale correct_size{2, 2, 1, 1};
  shared_ptr<float> input_ptr(new float[input_size.Prod()], [](float* ptr) { delete[] ptr; });
  memcpy(input_ptr.get(), input_raw, input_size.Prod() * sizeof(float));
  ms.SetDevice(gpu_device);
  ImageBatch input = NArray::MakeNArray(input_size, input_ptr);
  PoolingInfo pooling_info(PoolingInfo::Algorithm::kMax, 3, 3, 1, 1);
  ImageBatch output = Convolution::PoolingForward(input, pooling_info);
  auto output_ptr = output.Get();
  EXPECT_EQ(output.Size(), correct_size);
  for (int i = 0; i < correct_size.Prod(); ++i) {
    EXPECT_NEAR(output_ptr.get()[i], correct_raw[i], 0.001);
  }
}

TEST(PoolingForward, GpuWithExactPadding) {
  float input_raw[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  float correct_raw[] = {6, 7, 8, 8, 10, 11, 12, 12, 14, 15, 16, 16, 14, 15, 16, 16};
  auto& ms = MinervaSystem::Instance();
  Scale input_size{4, 4, 1, 1};
  Scale correct_size{4, 4, 1, 1};
  shared_ptr<float> input_ptr(new float[input_size.Prod()], [](float* ptr) { delete[] ptr; });
  memcpy(input_ptr.get(), input_raw, input_size.Prod() * sizeof(float));
  ms.SetDevice(gpu_device);
  ImageBatch input = NArray::MakeNArray(input_size, input_ptr);
  PoolingInfo pooling_info(PoolingInfo::Algorithm::kMax, 3, 3, 1, 1, 1, 1);
  ImageBatch output = Convolution::PoolingForward(input, pooling_info);
  auto output_ptr = output.Get();
  EXPECT_EQ(output.Size(), correct_size);
  for (int i = 0; i < correct_size.Prod(); ++i) {
    EXPECT_NEAR(output_ptr.get()[i], correct_raw[i], 0.001);
  }
}

TEST(PoolingForward, GpuWithInsufficientPadding) {
  float input_raw[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  float correct_raw[] = {6, 8, 8, 14, 16, 16, 14, 16, 16};
  auto& ms = MinervaSystem::Instance();
  Scale input_size{4, 4, 1, 1};
  Scale correct_size{3, 3, 1, 1};
  shared_ptr<float> input_ptr(new float[input_size.Prod()], [](float* ptr) { delete[] ptr; });
  memcpy(input_ptr.get(), input_raw, input_size.Prod() * sizeof(float));
  ms.SetDevice(gpu_device);
  ImageBatch input = NArray::MakeNArray(input_size, input_ptr);
  PoolingInfo pooling_info(PoolingInfo::Algorithm::kMax, 3, 3, 2, 2, 1, 1);
  ImageBatch output = Convolution::PoolingForward(input, pooling_info);
  auto output_ptr = output.Get();
  EXPECT_EQ(output.Size(), correct_size);
  for (int i = 0; i < correct_size.Prod(); ++i) {
    EXPECT_NEAR(output_ptr.get()[i], correct_raw[i], 0.001);
  }
}

TEST(PoolingForward, GpuWithTooMuchPadding) {
  float input_raw[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  float correct_raw[] = {6, 8, 14, 16};
  auto& ms = MinervaSystem::Instance();
  Scale input_size{4, 4, 1, 1};
  Scale correct_size{2, 2, 1, 1};
  shared_ptr<float> input_ptr(new float[input_size.Prod()], [](float* ptr) { delete[] ptr; });
  memcpy(input_ptr.get(), input_raw, input_size.Prod() * sizeof(float));
  ms.SetDevice(gpu_device);
  ImageBatch input = NArray::MakeNArray(input_size, input_ptr);
  PoolingInfo pooling_info(PoolingInfo::Algorithm::kMax, 4, 4, 3, 3, 2, 2);
  ImageBatch output = Convolution::PoolingForward(input, pooling_info);
  auto output_ptr = output.Get();
  EXPECT_EQ(output.Size(), correct_size);
  for (int i = 0; i < correct_size.Prod(); ++i) {
    EXPECT_NEAR(output_ptr.get()[i], correct_raw[i], 0.001);
  }
}
#endif
