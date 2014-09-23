#include <iostream>
#include <minerva.h>
#include <gtest/gtest.h>

using namespace minerva;
using namespace std;

TEST(Reduction, MaxOnFirstDimension) {
  Scale size{5, 3};
  shared_ptr<float> data( new float[size.Prod()] );
  for (int i = 0; i < size.Prod(); ++i) {
    data.get()[i] = i;
  }
  NArray na = NArray::MakeNArray(size, data, {1, 1});
  NArray na2 = na.Max(0);
  float* res = na2.Get();
  for (int i = 0; i < na2.Size().Prod(); ++i) {
    EXPECT_FLOAT_EQ(res[i], 5 * i + 4);
  }
}

TEST(Reduction, MaxOnSecondDimension) {
  Scale size{5, 3};
  shared_ptr<float> data(new float[size.Prod()]);
  for (int i = 0; i < size.Prod(); ++i) {
    data.get()[i] = i;
  }
  NArray na = NArray::MakeNArray(size, data, {1, 1});
  NArray na2 = na.Max(1);
  float* res = na2.Get();
  for (int i = 0; i < na2.Size().Prod(); ++i) {
    EXPECT_FLOAT_EQ(res[i], 10 + i);
  }
}

TEST(Reduction, SumOnFirstDimension) {
  Scale size{5, 3};
  shared_ptr<float> data(new float[size.Prod()]);
  for (int i = 0; i < size.Prod(); ++i) {
    data.get()[i] = i;
  }
  NArray na = NArray::MakeNArray(size, data, {1, 1});
  NArray na2 = na.Sum(0);
  float* res = na2.Get();
  for (int i = 0; i < na2.Size().Prod(); ++i) {
    EXPECT_FLOAT_EQ(res[i], 25 * i + 10);
  }
}

TEST(Reduction, SumOnSecondDimension) {
  Scale size{5, 3};
  shared_ptr<float> data(new float[size.Prod()]);
  for (int i = 0; i < size.Prod(); ++i) {
    data.get()[i] = i;
  }
  NArray na = NArray::MakeNArray(size, data, {1, 1});
  NArray na2 = na.Sum(1);
  float* res = na2.Get();
  for (int i = 0; i < na2.Size().Prod(); ++i) {
    EXPECT_FLOAT_EQ(res[i], 3 * i + 15);
  }
}
