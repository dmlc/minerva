#include "./minerva_utils.h"
#include <iostream>

namespace athena {

uint64_t CreateCpuDevice() {
  auto&& ms = minerva::MinervaSystem::Instance();
  return ms.device_manager().CreateCpuDevice();
}

uint64_t CreateGpuDevice(int id) {
  auto&& ms = minerva::MinervaSystem::Instance();
  return ms.device_manager().CreateGpuDevice(id);
}

int GetGpuDeviceCount() {
  auto&& ms = minerva::MinervaSystem::Instance();
  return ms.device_manager().GetGpuDeviceCount();
}

void WaitForAll() {
  auto&& ms = minerva::MinervaSystem::Instance();
  ms.backend().WaitForAll();
}

void SetDevice(uint64_t id) {
  auto&& ms = minerva::MinervaSystem::Instance();
  ms.current_device_id_ = id;
}

}  // namespace athena
