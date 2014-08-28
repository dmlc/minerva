#include <iostream>
#include <string>
#include <minerva.h>
#include <gtest/gtest.h>
#include <dag/dag.h>

using namespace std;

namespace minerva {

template<> class Inspector<MinervaSystem> {
  void GeneratePhysicalDag(const vector<uint64_t>& lids) {
    MinervaSystem::Instance().GeneratePhysicalDag(lids);
  }
};

}

using namespace minerva;

TEST(DevicePassingTest, Basic) {
  MinervaSystem& ms = MinervaSystem::Instance();
  EXPECT_EQ(ms.GetDeviceInfo().id, 0);

  DeviceInfo di1 = ms.CreateGPUDevice(0);
  ms.SetDevice(di1);
  EXPECT_EQ(ms.GetDeviceInfo().id, 1);
  EXPECT_EQ(ms.GetDeviceInfo().GPUList.size(), 1);
  EXPECT_EQ(ms.GetDeviceInfo().numStreams[0], 1);

  DeviceInfo di2 = ms.CreateGPUDevice(1, 2);
  ms.SetDevice(di2);
  EXPECT_EQ(ms.GetDeviceInfo().id, 2);
  EXPECT_EQ(ms.GetDeviceInfo().GPUList.size(), 1);
  EXPECT_EQ(ms.GetDeviceInfo().numStreams[0], 2);
}

TEST(DevicePassingTest, PassingThroughDag) {
  MinervaSystem& ms = MinervaSystem::Instance();
  NArray x = NArray::Randn({2, 4}, 0.0, 1.0, {1, 2});
  NArray y = NArray::Randn({4, 6}, 0.0, 1.0, {2, 2});
  NArray z = NArray::Randn({2, 6}, 0.0, 1.0, {1, 2});
  NArray t = x * y;

  DeviceInfo di = ms.CreateGPUDevice(0);
  ms.SetDevice(di);
  NArray s = t + z;
  s.Eval();

  PhysicalDag& pdag = ms.physical_dag();
  cout << pdag.NumNodes() << endl;
}

