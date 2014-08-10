#include <minerva.h>
#include <gtest/gtest.h>

using namespace minerva;
using namespace std;

TEST(GCTest, EvalInLoop) {
  MinervaSystem& ms = MinervaSystem::Instance();
  NArray narr = NArray::Constant({10, 8}, 0.0, {2, 2});
  for(int i = 0; i < 10; ++i) {
    narr += 1;
    narr.Eval();
    EXPECT_EQ(ms.logical_dag().NumNodes(), 3) << "wrong #logical_nodes in iter#" << i;
    if(i == 0)
      EXPECT_EQ(ms.physical_dag().NumNodes(), 16) << "wrong #physical_nodes in iter#" << i;
    else
      EXPECT_EQ(ms.physical_dag().NumNodes(), 12) << "wrong #physical_nodes in iter#" << i;
    EXPECT_EQ(ms.data_store().GetTotalBytes(DataStore::CPU), 320) << "wrong memory usage in iter#" << i;
    //cout << MinervaSystem::Instance().logical_dag().PrintDag() << endl;
    //cout << MinervaSystem::Instance().physical_dag().PrintDag() << endl;
  }
  float* val = narr.Get();
  for(int i = 0; i < 80; ++i)
    EXPECT_EQ(val[i], 10) << "value mismatch at i=" << i;
}
