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
  NArray a = NArray::Zeros({250, 500}, {1, 1});
  NArray b = NArray::Zeros({500, 400}, {1, 1});
  NArray c = a * b; // 250x400
  cout << "Call async eval" << endl;
  c.EvalAsync();
  cout << "Call eval end" << endl;
  NArray d = c + 1; // 250x400

  DeviceInfo di = ms.CreateCPUDevice();
  ms.set_device_info(di);

  NArray e = b * d.Trans(); // 500x250
  MinervaSystem::Instance().WaitForEvalFinish();
  cout << "Call sync eval" << endl;
  float* eptr = e.Get();
  for (int i = 0; i < 500 * 250; ++i) {
    ASSERT_EQ(eptr[i], 0.0);
  }
  delete [] eptr;
  cout << "Call eval end" << endl;
}

