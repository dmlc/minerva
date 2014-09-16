#include <iostream>
#include <string>
#include <minerva.h>
#include <gtest/gtest.h>
#include <dag/dag.h>
#include <common/inspector.h>
#include <device/device.h>
#include <device/device_info.h>

using namespace std;

namespace minerva {

template<> class Inspector<MinervaSystem> {
  public: void GeneratePhysicalDag(const vector<uint64_t>& lids) {
    MinervaSystem::Instance().GeneratePhysicalDag(lids);
  }
  public: uint64_t GetNodeID(const NArray na) {
    return na.data_node_->node_id();
  }
  public: DeviceInfo device_info(uint64_t device_id) {
    return MinervaSystem::Instance().GetDevice(device_id)->device_info_;
  }
};

}

using namespace minerva;

TEST(DevicePassingTest, Basic) {
  MinervaSystem& ms = MinervaSystem::Instance();
  EXPECT_EQ(ms.device_id(), 0);

  uint64_t id1 = ms.CreateGPUDevice(0);
  ms.set_device_id(id1);
  EXPECT_EQ(ms.device_id(), 1);
  EXPECT_EQ(Inspector<MinervaSystem>().device_info(1).gpu_list.size(), 1);
  EXPECT_EQ(Inspector<MinervaSystem>().device_info(1).num_streams[0], 1);

  uint64_t id2 = ms.CreateGPUDevice(1, 2);
  ms.set_device_id(id2);
  EXPECT_EQ(ms.device_id(), 2);
  EXPECT_EQ(Inspector<MinervaSystem>().device_info(2).gpu_list.size(), 1);
  EXPECT_EQ(Inspector<MinervaSystem>().device_info(2).num_streams[0], 2);
}

TEST(DevicePassingTest, PassingThroughDag1) {
  MinervaSystem& ms = MinervaSystem::Instance();
  ms.set_device_id(0);

  NArray x = NArray::Randn({2, 4}, 0.0, 1.0, {1, 1});
  NArray y = NArray::Randn({4, 6}, 0.0, 1.0, {1, 1});
  NArray t = x * y;

  uint64_t id = ms.CreateGPUDevice(0);
  ms.set_device_id(id);
  NArray z = NArray::Randn({2, 6}, 0.0, 1.0, {1, 1});
  NArray s = t + z;

  std::vector<uint64_t> lid_to_eval;
  lid_to_eval.push_back(Inspector<MinervaSystem>().GetNodeID(s));
  Inspector<MinervaSystem>().GeneratePhysicalDag(lid_to_eval);

  PhysicalDag& pdag = ms.physical_dag();
  EXPECT_EQ(dynamic_cast<PhysicalDataNode*>(pdag.GetNode(0))->data_.device_id, 0);
  EXPECT_EQ(dynamic_cast<PhysicalOpNode*>(pdag.GetNode(1))->op_.compute_fn->device_id, 0);
  EXPECT_EQ(dynamic_cast<PhysicalDataNode*>(pdag.GetNode(2))->data_.device_id, 0);
  EXPECT_EQ(dynamic_cast<PhysicalOpNode*>(pdag.GetNode(3))->op_.compute_fn->device_id, 0);
  EXPECT_EQ(dynamic_cast<PhysicalDataNode*>(pdag.GetNode(4))->data_.device_id, id);
  EXPECT_EQ(dynamic_cast<PhysicalOpNode*>(pdag.GetNode(5))->op_.compute_fn->device_id, id);
  EXPECT_EQ(dynamic_cast<PhysicalDataNode*>(pdag.GetNode(6))->data_.device_id, 0);
  EXPECT_EQ(dynamic_cast<PhysicalOpNode*>(pdag.GetNode(7))->op_.compute_fn->device_id, 0);
  EXPECT_EQ(dynamic_cast<PhysicalDataNode*>(pdag.GetNode(8))->data_.device_id, id);
  EXPECT_EQ(dynamic_cast<PhysicalOpNode*>(pdag.GetNode(9))->op_.compute_fn->device_id, id);
}

TEST(DevicePassingTest, PassingThroughDag2) {
  MinervaSystem& ms = MinervaSystem::Instance();
  ms.set_device_id(0);

  NArray x = NArray::Randn({2, 4}, 0.0, 1.0, {1, 1});
  NArray y = NArray::Randn({2, 4}, 0.0, 1.0, {1, 1});
  NArray s = x - y;

  uint64_t id1 = ms.CreateGPUDevice(0);
  ms.set_device_id(id1);
  NArray z = NArray::Randn({4, 8}, 0.0, 1.0, {1, 1});
  NArray w = NArray::Randn({4, 8}, 0.0, 1.0, {1, 1});
  NArray t = z + w;

  uint64_t id2 = ms.CreateGPUDevice(1, 2);
  ms.set_device_id(id2);
  NArray r = s * t;

  std::vector<uint64_t> lid_to_eval;
  lid_to_eval.push_back(Inspector<MinervaSystem>().GetNodeID(r));
  Inspector<MinervaSystem>().GeneratePhysicalDag(lid_to_eval);

  PhysicalDag& pdag = ms.physical_dag();
  EXPECT_EQ(dynamic_cast<PhysicalDataNode*>(pdag.GetNode(10))->data_.device_id, 0);
  EXPECT_EQ(dynamic_cast<PhysicalOpNode*>(pdag.GetNode(11))->op_.compute_fn->device_id, 0);
  EXPECT_EQ(dynamic_cast<PhysicalDataNode*>(pdag.GetNode(12))->data_.device_id, 0);
  EXPECT_EQ(dynamic_cast<PhysicalOpNode*>(pdag.GetNode(13))->op_.compute_fn->device_id, 0);
  EXPECT_EQ(dynamic_cast<PhysicalDataNode*>(pdag.GetNode(14))->data_.device_id, id1);
  EXPECT_EQ(dynamic_cast<PhysicalOpNode*>(pdag.GetNode(15))->op_.compute_fn->device_id, id1);
  EXPECT_EQ(dynamic_cast<PhysicalDataNode*>(pdag.GetNode(16))->data_.device_id, id1);
  EXPECT_EQ(dynamic_cast<PhysicalOpNode*>(pdag.GetNode(17))->op_.compute_fn->device_id, id1);
  EXPECT_EQ(dynamic_cast<PhysicalDataNode*>(pdag.GetNode(18))->data_.device_id, 0);
  EXPECT_EQ(dynamic_cast<PhysicalOpNode*>(pdag.GetNode(19))->op_.compute_fn->device_id, 0);
  EXPECT_EQ(dynamic_cast<PhysicalDataNode*>(pdag.GetNode(20))->data_.device_id, id1);
  EXPECT_EQ(dynamic_cast<PhysicalOpNode*>(pdag.GetNode(21))->op_.compute_fn->device_id, id1);
  EXPECT_EQ(dynamic_cast<PhysicalDataNode*>(pdag.GetNode(22))->data_.device_id, id2);
  EXPECT_EQ(dynamic_cast<PhysicalOpNode*>(pdag.GetNode(23))->op_.compute_fn->device_id, id2);
}

TEST(DevicePassingTest, DeviceFactory) {
  MinervaSystem& ms = MinervaSystem::Instance();

  uint64_t id1 = ms.CreateGPUDevice(0);
  ms.set_device_id(id1);
  EXPECT_EQ(ms.GetDevice(id1)->GetInfo().gpu_list.size(), 1);

  uint64_t id2 = ms.CreateGPUDevice(1, 2);
  ms.set_device_id(id2);
  EXPECT_EQ(ms.GetDevice(id2)->GetInfo().gpu_list.size(), 1);
  EXPECT_EQ(ms.GetDevice(id2)->GetInfo().num_streams[0], 2);
}

