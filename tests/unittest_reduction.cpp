#include <iostream>
#include <minerva.h>
#include <gtest/gtest.h>

using namespace minerva;
using namespace std;

TEST(Reduction, MaxOnFirstDimension) {
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

TEST(Reduction, MaxOnSecondDimension) {
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

TEST(Reduction, SumOnFirstDimension) {
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

TEST(Reduction, SumOnSecondDimension) {
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

