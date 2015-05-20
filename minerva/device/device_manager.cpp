#include "device_manager.h"
#include <dmlc/logging.h>
#include "device/device.h"
#include "common/cuda_utils.h"
#include "common/common.h"
#ifdef HAS_CUDA
#include <cuda.h>
#endif

using namespace std;

namespace minerva {

DeviceManager::DeviceManager() { }

DeviceManager::~DeviceManager() {
  for (auto i : device_storage_) {
    delete i.second;
  }
}

uint64_t DeviceManager::CreateCpuDevice() {
  auto id = GenerateDeviceId();
  Device* d = new CpuDevice(id, listener_);
  CHECK(device_storage_.emplace(id, d).second);
  return id;
}

uint64_t DeviceManager::CreateGpuDevice(int gid) {
#ifdef HAS_CUDA
  auto id = GenerateDeviceId();
  Device* d = new GpuDevice(id, listener_, gid);
  CHECK(device_storage_.emplace(id, d).second);
  return id;
#else
  common::FatalError("please recompile with macro HAS_CUDA");
#endif
}

int DeviceManager::GetGpuDeviceCount() {
#ifdef _HAS_CUDA
  int ret;
  CUDA_CALL(cudaGetDeviceCount(&ret));
  return ret;
#else
  common::FatalError("please recompile with macro HAS_CUDA");
#endif
}

Device* DeviceManager::GetDevice(uint64_t id) {
  return device_storage_.at(id);
}

void DeviceManager::FreeData(uint64_t id) {
  for (auto i : device_storage_) {
    i.second->FreeDataIfExist(id);
  }
}

uint64_t DeviceManager::GenerateDeviceId() {
  static uint64_t index_counter = 0;
  return index_counter++;
}

}  // namespace minerva

