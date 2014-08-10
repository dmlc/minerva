#include "common/nvector.h"
#include <iostream>
#include <gtest/gtest.h>

using namespace std;
using namespace minerva;

TEST(NVectorTest, IndexAccess) {
  std::vector<int> in;
  for(int i = 0; i < 12; ++i) {
    in.push_back(i);
  }
  NVector<int> nv(in, {3, 4});
  int rst = 0;
  for(int j = 0; j < 4; ++j)
    for(int i = 0; i < 3; ++i) {
      int val = nv[{i, j}];
      EXPECT_EQ(val, rst++) << "different value at {" << i << "," << j << "}";
    }
}

TEST(NVectorTest, MapFunction) {
  std::vector<int> in, out;
  for(int i = 0; i < 12; ++i) {
    in.push_back(i);
    out.push_back(i+1);
  }
  NVector<int> nv1(in, {3, 4});
  NVector<int> nv2 = nv1.Map<int>(
      [](const int& v)->int { return v+1; }
  );
  NVector<int> true_nv(out, {3, 4});
  EXPECT_EQ(nv2, true_nv);
}

TEST(NVectorTest, EquallySplitAndMerge) {
  Scale s1 = {4, 5};
  Scale s2 = {3, 3};
  cout << "s1=" << s1 << endl;
  cout << "s2=" << s2 << endl;
  NVector<Scale> s3 = s1.EquallySplit(s2);
  EXPECT_EQ(s2, s3.Size()) << "#partitions after EquallySplit is wrong";
  cout << "s1.EquallySplit(s2)=" << endl;
  cout << "size=" << s3.Size() << endl;
  cout << "content=" << endl;
  for(int i = 0; i < s3.Size()[0]; ++i) {
    cout << "|| ";
    for(int j = 0; j < s3.Size()[1]; ++j) {
      cout << s3[{i, j}] << " || ";
    }
    cout << endl;
  }
  EXPECT_EQ(s1, Scale::Merge(s3)) << "merged size is different";
}
