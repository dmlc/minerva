#include <minerva.h>
#include <gtest/gtest.h>

using namespace minerva;

uint64_t cpuDevice;
uint64_t gpuDevice;

class MinervaTestEnvironment : public testing::Environment {
 public:
  MinervaTestEnvironment(int* argc, char*** argv): argc(argc), argv(argv) {
  }
  void SetUp() {
    MinervaSystem::Instance().Initialize(argc, argv);
    cpuDevice = MinervaSystem::Instance().CreateCPUDevice();
    gpuDevice = MinervaSystem::Instance().CreateGPUDevice(0);
  }
  void TearDown() {
    MinervaSystem::Instance().Finalize();
  }
 private:
  int* argc;
  char*** argv;
};

int main(int argc, char** argv) {
  testing::AddGlobalTestEnvironment(new MinervaTestEnvironment(&argc, &argv));
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
