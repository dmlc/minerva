#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <cstdio>
#include "device_factory.h"

using namespace std;

namespace minerva {

DeviceFactory::DeviceFactory() {
  allocated_ = 1;
  device_storage_.clear();
  DeviceInfo dft;
  dft.id = 0;
  dft.cpu_list.push_back("localhost");
  InsertCPUDevice(dft);
}

void DeviceFactory::Reset() {
  allocated_ = 0;
}

void DeviceFactory::PrintDevice(DeviceInfo device_info) {
  printf("device id: %" PRIu64 "\n", device_info.id);
}

DeviceInfo DeviceFactory::DefaultInfo() {
  DeviceInfo result;
  result.id = 0;
  result.cpu_list.push_back("localhost");
  return result;
}

DeviceInfo DeviceFactory::CreateGPUDevice(int gid) {
  DeviceInfo result;
  result.id = allocated_++;
  result.gpu_list.push_back(gid);
  result.num_streams.push_back(1);
  InsertGPUDevice(result);
  return result;
}

DeviceInfo DeviceFactory::CreateGPUDevice(int gid, int num_stream) {
  DeviceInfo result;
  result.id = allocated_++;
  result.gpu_list.push_back(gid);
  result.num_streams.push_back(num_stream);
  InsertGPUDevice(result);
  return result;
}

Device* DeviceFactory::GetDevice(uint64_t id) {
  if (device_storage_.find(id) != device_storage_.end())
    return device_storage_[id];
  else
    return NULL;
}

Device* DeviceFactory::GetDevice(DeviceInfo info) {
  if (device_storage_.find(info.id) != device_storage_.end())
    return device_storage_[info.id];
  else
    return NULL;
}

void DeviceFactory::InsertCPUDevice(DeviceInfo info) {
  Device *device = new CpuDevice(info);
  device_storage_[info.id] = device;
}

void DeviceFactory::InsertGPUDevice(DeviceInfo info) {
  Device *device = new GpuDevice(info);
  device_storage_[info.id] = device;
}

}

