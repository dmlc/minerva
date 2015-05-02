#include "unittest_main.h"

using namespace minerva;
using namespace std;

TEST(Reduction, CpuMaxOnFirstDimension) {
  MinervaSystem::Instance().SetDevice(cpu_device);
  Scale size{5, 3};
  shared_ptr<float> data(new float[size.Prod()], [](float* ptr) {
    delete[] ptr;
  });
  for (int i = 0; i < size.Prod(); ++i) {
    data.get()[i] = i;
  }
  NArray na = NArray::MakeNArray(size, data);
  NArray na2 = na.Max(0);
  auto res = na2.Get();
  auto res_ptr = res.get();
  for (int i = 0; i < na2.Size().Prod(); ++i) {
    EXPECT_FLOAT_EQ(res_ptr[i], 5 * i + 4);
  }
}

TEST(Reduction, CpuMaxOnSecondDimension) {
  MinervaSystem::Instance().SetDevice(cpu_device);
  Scale size{5, 3};
  shared_ptr<float> data(new float[size.Prod()], [](float* ptr) {
    delete[] ptr;
  });
  for (int i = 0; i < size.Prod(); ++i) {
    data.get()[i] = i;
  }
  NArray na = NArray::MakeNArray(size, data);
  NArray na2 = na.Max(1);
  auto res = na2.Get();
  auto res_ptr = res.get();
  for (int i = 0; i < na2.Size().Prod(); ++i) {
    EXPECT_FLOAT_EQ(res_ptr[i], 10 + i);
  }
}

TEST(Reduction, CpuSumOnFirstDimension) {
  MinervaSystem::Instance().SetDevice(cpu_device);
  Scale size{5, 3};
  shared_ptr<float> data(new float[size.Prod()], [](float* ptr) {
    delete[] ptr;
  });
  for (int i = 0; i < size.Prod(); ++i) {
    data.get()[i] = i;
  }
  NArray na = NArray::MakeNArray(size, data);
  NArray na2 = na.Sum(0);
  auto res = na2.Get();
  auto res_ptr = res.get();
  for (int i = 0; i < na2.Size().Prod(); ++i) {
    EXPECT_FLOAT_EQ(res_ptr[i], 25 * i + 10);
  }
}

TEST(Reduction, CpuSumOnSecondDimension) {
  MinervaSystem::Instance().SetDevice(cpu_device);
  Scale size{5, 3};
  shared_ptr<float> data(new float[size.Prod()], [](float* ptr) {
    delete[] ptr;
  });
  for (int i = 0; i < size.Prod(); ++i) {
    data.get()[i] = i;
  }
  NArray na = NArray::MakeNArray(size, data);
  NArray na2 = na.Sum(1);
  auto res = na2.Get();
  auto res_ptr = res.get();
  for (int i = 0; i < na2.Size().Prod(); ++i) {
    EXPECT_FLOAT_EQ(res_ptr[i], 3 * i + 15);
  }
}

#ifdef HAS_CUDA
TEST(Reduction, GpuMaxOnFirstDimension) {
  MinervaSystem::Instance().SetDevice(gpu_device);
  Scale size{5, 3};
  shared_ptr<float> data(new float[size.Prod()], [](float* ptr) {
    delete[] ptr;
  });
  for (int i = 0; i < size.Prod(); ++i) {
    data.get()[i] = i;
  }
  NArray na = NArray::MakeNArray(size, data);
  NArray na2 = na.Max(0);
  auto res = na2.Get();
  auto res_ptr = res.get();
  for (int i = 0; i < na2.Size().Prod(); ++i) {
    EXPECT_FLOAT_EQ(res_ptr[i], 5 * i + 4);
  }
}

TEST(Reduction, GpuMaxOnSecondDimension) {
  MinervaSystem::Instance().SetDevice(gpu_device);
  Scale size{5, 3};
  shared_ptr<float> data(new float[size.Prod()], [](float* ptr) {
    delete[] ptr;
  });
  for (int i = 0; i < size.Prod(); ++i) {
    data.get()[i] = i;
  }
  NArray na = NArray::MakeNArray(size, data);
  NArray na2 = na.Max(1);
  auto res = na2.Get();
  auto res_ptr = res.get();
  for (int i = 0; i < na2.Size().Prod(); ++i) {
    EXPECT_FLOAT_EQ(res_ptr[i], 10 + i);
  }
}

TEST(Reduction, GpuSumOnFirstDimension) {
  MinervaSystem::Instance().SetDevice(gpu_device);
  Scale size{5, 3};
  shared_ptr<float> data(new float[size.Prod()], [](float* ptr) {
    delete[] ptr;
  });
  for (int i = 0; i < size.Prod(); ++i) {
    data.get()[i] = i;
  }
  NArray na = NArray::MakeNArray(size, data);
  NArray na2 = na.Sum(0);
  auto res = na2.Get();
  auto res_ptr = res.get();
  for (int i = 0; i < na2.Size().Prod(); ++i) {
    EXPECT_FLOAT_EQ(res_ptr[i], 25 * i + 10);
  }
}

TEST(Reduction, GpuSumOnSecondDimension) {
  MinervaSystem::Instance().SetDevice(gpu_device);
  Scale size{5, 3};
  shared_ptr<float> data(new float[size.Prod()], [](float* ptr) {
    delete[] ptr;
  });
  for (int i = 0; i < size.Prod(); ++i) {
    data.get()[i] = i;
  }
  NArray na = NArray::MakeNArray(size, data);
  NArray na2 = na.Sum(1);
  auto res = na2.Get();
  auto res_ptr = res.get();
  for (int i = 0; i < na2.Size().Prod(); ++i) {
    EXPECT_FLOAT_EQ(res_ptr[i], 3 * i + 15);
  }
}
#endif
