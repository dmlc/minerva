#include <minerva.h>
#include <gtest/gtest.h>

using namespace minerva;
using namespace std;

TEST(GCCorrectness, EvalInLoop) {
  MinervaSystem& ms = MinervaSystem::Instance();
  NArray narr = NArray::Constant({10, 8}, 0.0);
  for(int i = 0; i < 10; ++i) {
    narr += 1;
    ms.backend().WaitForAll();
    EXPECT_EQ(ms.physical_dag().NumNodes(), 1) << "wrong #physical_nodes in iter#" << i;
  }
  shared_ptr<float> val = narr.Get();
  for (int i = 0; i < 80; ++i) {
    ASSERT_EQ(val.get()[i], 10) << "value mismatch at i=" << i;
  }
}

TEST(GCCorrectness, EvalPartial) {
  MinervaSystem& ms = MinervaSystem::Instance();
  NArray a = NArray::Constant({10, 8}, 0.0);
  vector<NArray> arr;
  for (int i = 0; i < 10; ++i) {
    arr.push_back(a + 1);
  }
  ms.backend().WaitForAll();
  for(size_t i = 0; i < arr.size(); ++i) {
    ASSERT_EQ(ms.physical_dag().NumNodes(), arr.size() + 1 - i);
    arr[i] = NArray();
    ms.backend().WaitForAll();
  }
}

TEST(GCCorrectness, ChangeInternRCAfterEval) {
  MinervaSystem& ms = MinervaSystem::Instance();
  NArray a = NArray::Constant({10, 8}, 0.0);
  ms.backend().WaitForAll();
  EXPECT_EQ(ms.physical_dag().NumNodes(), 1);
  NArray b = a + 1;
  NArray c = a + 1;
  ms.backend().WaitForAll();
  EXPECT_EQ(ms.physical_dag().NumNodes(), 3);
  b = NArray();
  ms.backend().WaitForAll();
  EXPECT_EQ(ms.physical_dag().NumNodes(), 2);
}

TEST(GCCorrectness, ChangeExternRCAfterEval) {
  MinervaSystem& ms = MinervaSystem::Instance();
  NArray a = NArray::Constant({10, 8}, 0.0);
  {
    NArray b = NArray::Constant({10, 8}, 0.0);
    ms.backend().WaitForAll();
    EXPECT_EQ(ms.physical_dag().NumNodes(), 2);
  }
  ms.backend().WaitForAll();
  EXPECT_EQ(ms.physical_dag().NumNodes(), 1);
}

TEST(GCCorrectness, ChangeBothRCAfterEval) {
  MinervaSystem& ms = MinervaSystem::Instance();
  NArray a, b;
  {
    NArray c = NArray::Constant({10, 8}, 0.0);
    a = c + 1;
    b = c + 2;
  }
  ms.backend().WaitForAll();
  EXPECT_EQ(ms.physical_dag().NumNodes(), 2);
  // Check correctness
  shared_ptr<float> aptr = a.Get();
  for (int i = 0; i < 80; ++i) {
    ASSERT_EQ(aptr.get()[i], 1);
  }
  shared_ptr<float> bptr = b.Get();
  for (int i = 0; i < 80; ++i) {
    ASSERT_EQ(bptr.get()[i], 2);
  }
}
