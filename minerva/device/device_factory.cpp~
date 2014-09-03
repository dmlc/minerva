#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <stdio.h>
#include "device_factory.h"

using namespace std;

namespace minerva {

void DeviceFactory::Reset() {
  allocated_ = 0;
}

void DeviceFactory::PrintDevice(DeviceInfo device_info) {
  printf("device id: %" PRIu64 "\n", device_info.id);
}

DeviceInfo DeviceFactory::DefaultInfo() {
  DeviceInfo result;
  result.id = 0;
  if (allocated_ == 0) {
    ++allocated_;
  }
  result.cpu_list.push_back("localhost");
  InsertGPUDevice(result);
  return result;
}

DeviceInfo DeviceFactory::GPUDeviceInfo(int gid) {
  DeviceInfo result;
  result.id = allocated_++;
  result.gpu_list.push_back(gid);
  result.num_streams.push_back(1);
  InsertGPUDevice(result);
  return result;
}

DeviceInfo DeviceFactory::GPUDeviceInfo(int gid, int num_stream) {
  DeviceInfo result;
  result.id = allocated_++;
  result.gpu_list.push_back(gid);
  result.num_streams.push_back(num_stream);
  InsertGPUDevice(result);
  return result;
}

Device DeviceFactory::GetDevice(uint64_t id) {
  return device_storage_[id];
}

Device DeviceFactory::GetDevice(DeviceInfo info) {
  return device_storage_[info.id];
}

void DeviceFactory::InsertGPUDevice(DeviceInfo info) {
  Device device(info);
  device_storage_[info.id] = device;
}

}

