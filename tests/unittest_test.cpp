#include <iostream>
#include <string>
#include <minerva.h>
#include <gtest/gtest.h>
#include <dag/dag.h>
#include <common/inspector.h>
#include <device/device.h>

using namespace std;

using namespace minerva;

TEST(Test, DeviceDebug) {
  MinervaSystem& ms = MinervaSystem::Instance();
  DeviceFactory df = DeviceFactory::Instance();
  df.Reset();
  ms.set_device_info(df.DefaultInfo());

  NArray x = NArray::Randn({2, 4}, 0.0, 1.0, {1, 1});
  NArray y = NArray::Randn({4, 6}, 0.0, 1.0, {1, 1});
  NArray t = x * y;

  NArray z = NArray::Randn({2, 6}, 0.0, 1.0, {1, 1});
  NArray s = t + z;

  s.Eval();
}

