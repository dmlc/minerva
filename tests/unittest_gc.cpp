#include <minerva.h>
#include <gtest/gtest.h>
#include <device/data_store.h>
#include <common/inspector.h>

using namespace minerva;
using namespace std;

namespace minerva {

template<> class Inspector<Device> {
  public: DataStore* GetDataStore(uint64_t device_id) {
    return MinervaSystem::Instance().GetDevice(device_id)->data_store_;
  }
};

}

TEST(GCCorrectness, EvalInLoop) {
  MinervaSystem& ms = MinervaSystem::Instance();
  NArray narr = NArray::Constant({10, 8}, 0.0, {1, 1});
  for(int i = 0; i < 10; ++i) {
    narr += 1;
    //cout << ms.logical_dag().PrintDag<ExternRCPrinter>() << endl;
    //cout << ms.physical_dag().PrintDag() << endl;
    narr.Eval();
    EXPECT_EQ(ms.logical_dag().NumNodes(), 1) << "wrong #logical_nodes in iter#" << i;
    EXPECT_EQ(ms.physical_dag().NumNodes(), 1) << "wrong #physical_nodes in iter#" << i;
    EXPECT_EQ(Inspector<Device>().GetDataStore(0)->GetTotalBytes(DataStore::CPU), 320) << "wrong memory usage in iter#" << i;
    cout << "iter #" << i << " succeed!" << endl;
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
    ASSERT_EQ(ms.logical_dag().NumNodes(), 11);
    ASSERT_EQ(ms.physical_dag().NumNodes(), 20 - i);
    cout << "Eval #" << i << " succeed!" << endl;
  }
  //EXPECT_EQ(ms.data_store().GetTotalBytes(DataStore::CPU), 3520);
}

TEST(GCCorrectness, ChangeInternRCAfterEval) {
  MinervaSystem& ms = MinervaSystem::Instance();
  NArray a = NArray::Constant({10, 8}, 0.0, {1, 1});
  a.Eval();
  EXPECT_EQ(ms.physical_dag().NumNodes(), 1);
  //EXPECT_EQ(ms.data_store().GetTotalBytes(DataStore::CPU), 320);
  NArray b = a + 1;
  NArray c = a + 1;
  b.Eval();
  EXPECT_EQ(ms.logical_dag().NumNodes(), 3);
  EXPECT_EQ(ms.physical_dag().NumNodes(), 4);
  c.Eval();
  EXPECT_EQ(ms.logical_dag().NumNodes(), 3);
  EXPECT_EQ(ms.physical_dag().NumNodes(), 3);
}

TEST(GCCorrectness, ChangeExternRCAfterEval) {
  MinervaSystem& ms = MinervaSystem::Instance();
  NArray a = NArray::Constant({10, 8}, 0.0, {1, 1});
  {
    NArray b = NArray::Constant({10, 8}, 0.0, {1, 1});
    b.Eval();
    EXPECT_EQ(ms.logical_dag().NumNodes(), 2);
    EXPECT_EQ(ms.physical_dag().NumNodes(), 3);
  }
  a.Eval();
  EXPECT_EQ(ms.logical_dag().NumNodes(), 1);
  EXPECT_EQ(ms.physical_dag().NumNodes(), 1);
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
  //cout << ms.logical_dag().PrintDag() << endl;
  EXPECT_EQ(ms.logical_dag().NumNodes(), 2);
  EXPECT_EQ(ms.physical_dag().NumNodes(), 4);
  b.Eval();
  EXPECT_EQ(ms.logical_dag().NumNodes(), 2);
  EXPECT_EQ(ms.physical_dag().NumNodes(), 2);
  // check correctness
  float* aptr = a.Get();
  for(int i = 0; i < 80; ++i)
    ASSERT_EQ(aptr[i], 1);
  float* bptr = b.Get();
  for(int i = 0; i < 80; ++i)
    ASSERT_EQ(bptr[i], 2);
}
