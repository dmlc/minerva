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
  NArray a = NArray::Randn({250, 500}, 0, 1, {1, 1});
}

