#include <minerva.h>
#include <gtest/gtest.h>

using namespace minerva;
using namespace std;

TEST(GCCorrectness, EvalInLoop) {
  MinervaSystem& ms = MinervaSystem::Instance();
  NArray narr = NArray::Constant({10, 8}, 0.0);
  for(int i = 0; i < 10; ++i) {
    narr += 1;
    //cout << ms.physical_dag().PrintDag() << endl;
    narr.Eval();
    EXPECT_EQ(ms.physical_dag().NumNodes(), 1) << "wrong #physical_nodes in iter#" << i;
    //EXPECT_EQ(ms.data_store().GetTotalBytes(DataStore::CPU), 320) << "wrong memory usage in iter#" << i;
    cout << "iter #" << i << " succeed!" << endl;
  }
  shared_ptr<float> val = narr.Get();
  for(int i = 0; i < 80; ++i)
    ASSERT_EQ(val.get()[i], 10) << "value mismatch at i=" << i;
}

TEST(GCCorrectness, EvalPartial) {
  MinervaSystem& ms = MinervaSystem::Instance();
  NArray a = NArray::Constant({10, 8}, 0.0);
  vector<NArray> arr;
  for(int i = 0; i < 10; ++i)
    arr.push_back(a + 1);
  for(size_t i = 0; i < arr.size(); ++i) {
    arr[i].Eval();
    ASSERT_EQ(ms.physical_dag().NumNodes(), 20 - i);
    cout << "Eval #" << i << " succeed!" << endl;
  }
  //EXPECT_EQ(ms.data_store().GetTotalBytes(DataStore::CPU), 3520);
}

TEST(GCCorrectness, ChangeInternRCAfterEval) {
  MinervaSystem& ms = MinervaSystem::Instance();
  NArray a = NArray::Constant({10, 8}, 0.0);
  a.Eval();
  EXPECT_EQ(ms.physical_dag().NumNodes(), 1);
  //EXPECT_EQ(ms.data_store().GetTotalBytes(DataStore::CPU), 320);
  NArray b = a + 1;
  NArray c = a + 1;
  b.Eval();
  EXPECT_EQ(ms.physical_dag().NumNodes(), 4);
  c.Eval();
  EXPECT_EQ(ms.physical_dag().NumNodes(), 3);
}

TEST(GCCorrectness, ChangeExternRCAfterEval) {
  MinervaSystem& ms = MinervaSystem::Instance();
  NArray a = NArray::Constant({10, 8}, 0.0);
  {
    NArray b = NArray::Constant({10, 8}, 0.0);
    b.Eval();
    EXPECT_EQ(ms.physical_dag().NumNodes(), 3);
  }
  a.Eval();
  EXPECT_EQ(ms.physical_dag().NumNodes(), 1);
}

TEST(GCCorrectness, ChangeBothRCAfterEval) {
  MinervaSystem& ms = MinervaSystem::Instance();
  NArray a, b;
  {
    NArray c = NArray::Constant({10, 8}, 0.0);
    c.Eval();
    a = c + 1;
    b = c + 2;
  }
  a.Eval();
  //cout << ms.logical_dag().PrintDag() << endl;
  EXPECT_EQ(ms.physical_dag().NumNodes(), 4);
  b.Eval();
  EXPECT_EQ(ms.physical_dag().NumNodes(), 2);
  // check correctness
  shared_ptr<float> aptr = a.Get();
  for(int i = 0; i < 80; ++i)
    ASSERT_EQ(aptr.get()[i], 1);
  shared_ptr<float> bptr = b.Get();
  for(int i = 0; i < 80; ++i)
    ASSERT_EQ(bptr.get()[i], 2);
}
