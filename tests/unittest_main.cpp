#include "unittest_main.h"

uint64_t cpu_device;
#ifdef HAS_CUDA
uint64_t gpu_device;
std::vector<uint64_t> gpu_devices;
#endif

using namespace minerva;

class MinervaTestEnvironment final : public testing::Environment {
 public:
  MinervaTestEnvironment(int* argc, char*** argv) : argc(argc), argv(argv) {
  }
  void SetUp() override {
    MinervaSystem::Initialize(argc, argv);
    auto&& ms = MinervaSystem::Instance();
    cpu_device = ms.device_manager().CreateCpuDevice();
#ifdef HAS_CUDA
    gpu_device = ms.device_manager().CreateGpuDevice(0);
    gpu_devices = {gpu_device};
    for (int i = 1; i < ms.device_manager().GetGpuDeviceCount(); ++i) {
      gpu_devices.push_back(ms.device_manager().CreateGpuDevice(i));
    }
#endif
  }
  void TearDown() override {
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

