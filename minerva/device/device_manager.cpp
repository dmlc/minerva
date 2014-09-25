#include "device/device_manager.h"
#include "device/device.h"
#include <glog/logging.h>

using namespace std;

namespace minerva {

DeviceManager::DeviceManager(DeviceListener* l) : listener_(l) {
}

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

#ifdef HAS_CUDA

uint64_t DeviceManager::CreateGpuDevice(int gid) {
  auto id = GenerateDeviceId();
  Device* d = new GpuDevice(id, listener_, gid);
  CHECK(device_storage_.emplace(id, d).second);
  return id;
}

#endif

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

