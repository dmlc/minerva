#include <iostream>
#include <string>
#include <minerva.h>
#include <gtest/gtest.h>
#include <dag/dag.h>
#include <common/inspector.h>

using namespace std;

namespace minerva {

template<> class Inspector<MinervaSystem> {
  public: void GeneratePhysicalDag(const vector<uint64_t>& lids) {
    MinervaSystem::Instance().GeneratePhysicalDag(lids);
  }
  public: uint64_t GetNodeID(const NArray na) {
    return na.data_node_->node_id();
  }
};

}

using namespace minerva;

TEST(DevicePassingTest, Basic) {
  MinervaSystem& ms = MinervaSystem::Instance();
  DeviceFactory df = DeviceFactory::Instance();
  df.Reset();
  ms.SetDevice(df.default_info());
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

TEST(DevicePassingTest, PassingThroughDag1) {
  MinervaSystem& ms = MinervaSystem::Instance();
  DeviceFactory df = DeviceFactory::Instance();
  df.Reset();
  ms.SetDevice(df.default_info());

  NArray x = NArray::Randn({2, 4}, 0.0, 1.0, {1, 1});
  NArray y = NArray::Randn({4, 6}, 0.0, 1.0, {1, 1});
  NArray t = x * y;

  DeviceInfo di = ms.CreateGPUDevice(0);
  ms.SetDevice(di);
  NArray z = NArray::Randn({2, 6}, 0.0, 1.0, {1, 1});
  NArray s = t + z;

  std::vector<uint64_t> lid_to_eval;
  lid_to_eval.push_back(Inspector<MinervaSystem>().GetNodeID(s));
  Inspector<MinervaSystem>().GeneratePhysicalDag(lid_to_eval);

  PhysicalDag& pdag = ms.physical_dag();
  EXPECT_EQ(pdag.GetNode(0)->device_info().id, 0);
  EXPECT_EQ(pdag.GetNode(1)->device_info().id, 0);
  EXPECT_EQ(pdag.GetNode(2)->device_info().id, 0);
  EXPECT_EQ(pdag.GetNode(3)->device_info().id, 0);
  EXPECT_EQ(pdag.GetNode(4)->device_info().id, di.id);
  EXPECT_EQ(pdag.GetNode(5)->device_info().id, di.id);
  EXPECT_EQ(pdag.GetNode(6)->device_info().id, 0);
  EXPECT_EQ(pdag.GetNode(7)->device_info().id, 0);
  EXPECT_EQ(pdag.GetNode(8)->device_info().id, di.id);
  EXPECT_EQ(pdag.GetNode(9)->device_info().id, di.id);
}

TEST(DevicePassingTest, PassingThroughDag2) {
  MinervaSystem& ms = MinervaSystem::Instance();
  DeviceFactory df = DeviceFactory::Instance();
  df.Reset();
  ms.SetDevice(df.default_info());

  NArray x = NArray::Randn({2, 4}, 0.0, 1.0, {1, 1});
  NArray y = NArray::Randn({2, 4}, 0.0, 1.0, {1, 1});
  NArray s = x - y;

  DeviceInfo di1 = ms.CreateGPUDevice(0);
  ms.SetDevice(di1);
  NArray z = NArray::Randn({4, 8}, 0.0, 1.0, {1, 1});
  NArray w = NArray::Randn({4, 8}, 0.0, 1.0, {1, 1});
  NArray t = z + w;

  DeviceInfo di2 = ms.CreateGPUDevice(1, 2);
  ms.SetDevice(di2);
  NArray r = s * t;

  std::vector<uint64_t> lid_to_eval;
  lid_to_eval.push_back(Inspector<MinervaSystem>().GetNodeID(r));
  Inspector<MinervaSystem>().GeneratePhysicalDag(lid_to_eval);

  PhysicalDag& pdag = ms.physical_dag();
  EXPECT_EQ(pdag.GetNode(10)->device_info().id, 0);
  EXPECT_EQ(pdag.GetNode(11)->device_info().id, 0);
  EXPECT_EQ(pdag.GetNode(12)->device_info().id, 0);
  EXPECT_EQ(pdag.GetNode(13)->device_info().id, 0);
  EXPECT_EQ(pdag.GetNode(14)->device_info().id, di1.id);
  EXPECT_EQ(pdag.GetNode(15)->device_info().id, di1.id);
  EXPECT_EQ(pdag.GetNode(16)->device_info().id, di1.id);
  EXPECT_EQ(pdag.GetNode(17)->device_info().id, di1.id);
  EXPECT_EQ(pdag.GetNode(18)->device_info().id, 0);
  EXPECT_EQ(pdag.GetNode(19)->device_info().id, 0);
  EXPECT_EQ(pdag.GetNode(20)->device_info().id, di1.id);
  EXPECT_EQ(pdag.GetNode(21)->device_info().id, di1.id);
  EXPECT_EQ(pdag.GetNode(22)->device_info().id, di2.id);
  EXPECT_EQ(pdag.GetNode(23)->device_info().id, di2.id);
}

