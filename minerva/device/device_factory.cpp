#include "device_factory.h"
#include "device/device.h"
#include <glog/logging.h>

using namespace std;

namespace minerva {

DeviceFactory::DeviceFactory(DeviceListener* l) : listener_(l) {
}

DeviceFactory::~DeviceFactory() {
  for (auto i : device_storage_) {
    delete i.second;
  }
}

uint64_t DeviceFactory::CreateCpuDevice() {
  auto id = GenerateDeviceId();
  Device* d = new CpuDevice(id, listener_);
  CHECK(device_storage_.emplace(id, d).second);
  return id;
}

#ifdef HAS_CUDA

uint64_t DeviceFactory::CreateGpuDevice(int gid) {
  auto id = GenerateDeviceId();
  Device* d = new GpuDevice(id, listener_, gid);
  CHECK(device_storage_.emplace(id, id).second);
  return id;
}

#endif

Device* DeviceFactory::GetDevice(uint64_t id) {
  return device_storage_.at(id);
}

uint64_t DeviceFactory::GenerateDeviceId() {
  static uint64_t index_counter = 0;
  return index_counter++;
}

}  // namespace minerva

