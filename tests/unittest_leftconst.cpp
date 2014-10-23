#include "unittest_main.h"

using namespace minerva;
using namespace std;

TEST(LeftConst, LeftConst) {
  auto& ms = MinervaSystem::Instance();
  ms.current_device_id_ = cpu_device;
  int m = 8;
  int k = 100;
  NArray a = NArray::Randn({m, k}, 0.0, 1.0);
  ms.current_device_id_ = gpu_device;
  NArray b = 1 - a;
  NArray b1 = 1 - b;
  NArray b2 = 2 - b1;
  NArray b3 = 3 - b2;
  NArray b4 = 4 - b3;
  NArray b5 = 5 - b4;
  auto in = a.Get();
  auto res = b.Get();
  auto in_ptr = in.get();
  auto res_ptr = res.get();
  for (int i = 0; i < a.Size().Prod(); ++i) {
    EXPECT_FLOAT_EQ(1 - in_ptr[i], res_ptr[i]);
  }
}

