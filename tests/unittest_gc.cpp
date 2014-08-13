#include <minerva.h>
#include <gtest/gtest.h>

using namespace minerva;
using namespace std;

TEST(GCCorrectness, EvalInLoop) {
  MinervaSystem& ms = MinervaSystem::Instance();
  EXPECT_EQ(ms.data_store().GetTotalBytes(DataStore::CPU), 0);
  NArray narr = NArray::Constant({10, 8}, 0.0, {2, 2});
  for(int i = 0; i < 10; ++i) {
    narr += 1;
    //cout << ms.logical_dag().PrintDag<ExternRCPrinter>() << endl;
    //cout << ms.physical_dag().PrintDag() << endl;
    narr.Eval();
    EXPECT_EQ(ms.logical_dag().NumNodes(), 3) << "wrong #logical_nodes in iter#" << i;
    if(i == 0)
      EXPECT_EQ(ms.physical_dag().NumNodes(), 16) << "wrong #physical_nodes in iter#" << i;
    else
      EXPECT_EQ(ms.physical_dag().NumNodes(), 12) << "wrong #physical_nodes in iter#" << i;
    EXPECT_EQ(ms.data_store().GetTotalBytes(DataStore::CPU), 320) << "wrong memory usage in iter#" << i;
  }
  float* val = narr.Get();
  for(int i = 0; i < 80; ++i)
    ASSERT_EQ(val[i], 10) << "value mismatch at i=" << i;
}

TEST(GCCorrectness, ScopedInstance) {
  MinervaSystem& ms = MinervaSystem::Instance();
  EXPECT_EQ(ms.data_store().GetTotalBytes(DataStore::CPU), 0);
  NArray a = NArray::Constant({10, 8}, 0.0, {2, 2});
  {
    NArray b = NArray::Constant({10, 8}, 0.0, {2, 2});
    b.Eval();
    EXPECT_EQ(ms.logical_dag().NumNodes(), 2);
    EXPECT_EQ(ms.physical_dag().NumNodes(), 8);
  }
  a.Eval();
  EXPECT_EQ(ms.logical_dag().NumNodes(), 1);
  EXPECT_EQ(ms.physical_dag().NumNodes(), 8);
}
