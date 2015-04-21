#include "unittest_main.h"
#include <cmath>

using namespace std;
using namespace minerva;

TEST(sgemm, CpuSgemm) {
  auto& ms = MinervaSystem::Instance();
  ms.SetDevice(cpu_device);
  Scale sizeA{3, 2};
  Scale sizeB{2, 5};
  auto a = NArray::Randn(sizeA, 0, 5);
  auto b = NArray::Randn(sizeB, 0, 5);
  auto c = a * b;
  auto a_ptr = a.Get();
  auto b_ptr = b.Get();
  auto c_ptr = c.Get();
  Scale sizeC = c.Size();
  for (int i = 0; i < sizeC[0]; ++i) {
    for (int j = 0; j < sizeC[1]; ++j) {
      float sum = 0;
      for (int k = 0; k < sizeA[1]; ++k) {
        sum += a_ptr.get()[i + k * sizeA[0]] * b_ptr.get()[k + j * sizeB[0]];
      }
      EXPECT_FLOAT_EQ(c_ptr.get()[i + j * sizeC[0]], sum);
    }
  }
}

#ifdef HAS_CUDA
TEST(sgemm, CpuGpuCrossCheck) {
  auto& ms = MinervaSystem::Instance();
  ms.SetDevice(cpu_device);
  Scale sizeA{3, 2};
  Scale sizeB{2, 3};
  auto a = NArray::Randn(sizeA, 0, 5);
  auto b = NArray::Randn(sizeB, 0, 5);
  auto cc = a * b;
  ms.SetDevice(gpu_device);
  auto cg = a * b;

  auto cc_ptr = cc.Get();
  auto cg_ptr = cg.Get();
  Scale sizeC = cc.Size();
  for (int i = 0; i < sizeC.Prod(); ++i) {
    EXPECT_FLOAT_EQ(cc_ptr.get()[i], cg_ptr.get()[i]);
  }
}
#endif
