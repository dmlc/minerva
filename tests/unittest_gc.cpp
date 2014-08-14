#include <minerva.h>
#include <gtest/gtest.h>

using namespace minerva;
using namespace std;

TEST(GCCorrectness, DISABLED_EvalInLoop) {
  MinervaSystem& ms = MinervaSystem::Instance();
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

TEST(GCCorrectness, EvalPartial) {
  MinervaSystem& ms = MinervaSystem::Instance();
  NArray a = NArray::Constant({10, 8}, 0.0, {1, 1});
  vector<NArray> arr;
  for(int i = 0; i < 10; ++i)
    arr.push_back(a + 1);
  for(size_t i = 0; i < arr.size(); ++i) {
    arr[i].Eval();
    ASSERT_EQ(ms.logical_dag().NumNodes(), 21 - i);
    cout << "Eval #" << i << " succeed!" << endl;
  }
  //EXPECT_EQ(ms.data_store().GetTotalBytes(DataStore::CPU), 3520);
}

TEST(GCCorrectness, ChangeInternRCAfterEval) {
  MinervaSystem& ms = MinervaSystem::Instance();
  NArray a = NArray::Constant({10, 8}, 0.0, {1, 1});
  a.Eval();
  //EXPECT_EQ(ms.data_store().GetTotalBytes(DataStore::CPU), 320);
  NArray b = a + 1;
  NArray c = a + 1;
  b.Eval();
  c.Eval();
}

TEST(GCCorrectness, ChangeExternRCAfterEval) {
  MinervaSystem& ms = MinervaSystem::Instance();
  NArray a = NArray::Constant({10, 8}, 0.0, {2, 2});
  {
    NArray b = NArray::Constant({10, 8}, 0.0, {2, 2});
    b.Eval();
    EXPECT_EQ(ms.logical_dag().NumNodes(), 2);
    //EXPECT_EQ(ms.physical_dag().NumNodes(), 8);
  }
  a.Eval();
  EXPECT_EQ(ms.logical_dag().NumNodes(), 1);
  //EXPECT_EQ(ms.physical_dag().NumNodes(), 8);
}

TEST(GCCorrectness, ChangeBothRCAfterEval) {
  MinervaSystem& ms = MinervaSystem::Instance();
  NArray a, b;
  {
    NArray c = NArray::Constant({10, 8}, 0.0, {1, 1});
    c.Eval();
    a = c + 1;
    b = c + 2;
  }
  a.Eval();
  cout << ms.logical_dag().PrintDag() << endl;
  EXPECT_EQ(ms.logical_dag().NumNodes(), 4);
  b.Eval();
  EXPECT_EQ(ms.logical_dag().NumNodes(), 2);
}
