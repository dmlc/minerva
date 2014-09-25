#include <minerva.h>
#include <iostream>
#include <gtest/gtest.h>

using namespace std;
using namespace minerva;

TEST(ScaleTest, FlattenIndex) {
  Scale st = {0, 0}, ed = {4, 5};
  ScaleRange range = ScaleRange::MakeRange(st, ed);
  EXPECT_EQ(range.Flatten({0, 0}), 0);
  EXPECT_EQ(range.Flatten({3, 4}), 19);
  EXPECT_EQ(range.Flatten({0, 4}), 16);
  EXPECT_EQ(range.Flatten({3, 0}), 3);
  EXPECT_EQ(range.Flatten({1, 2}), 9);
  EXPECT_EQ(range.Flatten({3, 2}), 11);
}

TEST(ScaleTest, FlattenIndexWithNonOriginStart) {
  Scale st = {1, 1}, ed = {5, 6};
  ScaleRange range = ScaleRange::MakeRange(st, ed);
  EXPECT_EQ(range.Flatten({1, 1}), 0);
  EXPECT_EQ(range.Flatten({4, 5}), 19);
  EXPECT_EQ(range.Flatten({1, 5}), 16);
  EXPECT_EQ(range.Flatten({4, 1}), 3);
  EXPECT_EQ(range.Flatten({2, 3}), 9);
  EXPECT_EQ(range.Flatten({4, 3}), 11);
}

