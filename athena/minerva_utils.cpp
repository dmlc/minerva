#include "./minerva_utils.h"
#include <memory>
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

minerva::Scale ToScale(std::vector<int>* v) {
  minerva::Scale r(std::move(*v));
  return r;
}

std::vector<int> OfScale(minerva::Scale const& s) {
  std::vector<int> ret;
  for (auto i : s) {
    ret.push_back(i);
  }
  return ret;
}

}  // namespace athena
