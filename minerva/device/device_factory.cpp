#include "device_factory.h"

using namespace std;

namespace minerva {

DeviceFactory::DeviceFactory() {
  allocated_ = 1;
  device_storage_.clear();
  DeviceInfo dft;
  dft.cpu_list.push_back("localhost");
  InsertCPUDevice(0, dft);
}

void DeviceFactory::Reset() {
  allocated_ = 0;
}

uint64_t DeviceFactory::CreateCPUDevice() {
  DeviceInfo result;
  InsertCPUDevice(allocated_, result);
  return allocated_++;
}

uint64_t DeviceFactory::CreateGPUDevice(int gid) {
  DeviceInfo result;
  result.gpu_list.push_back(gid);
  result.num_streams.push_back(1);
  InsertGPUDevice(allocated_, result);
  return allocated_++;
}

uint64_t DeviceFactory::CreateGPUDevice(int gid, int num_stream) {
  DeviceInfo result;
  result.gpu_list.push_back(gid);
  result.num_streams.push_back(num_stream);
  InsertGPUDevice(allocated_, result);
  return allocated_++;
}

Device* DeviceFactory::GetDevice(uint64_t id) {
  if (device_storage_.find(id) != device_storage_.end())
    return device_storage_[id];
  else
    return NULL;
}

void DeviceFactory::InsertCPUDevice(uint64_t id, DeviceInfo info) {
  Device *device = new CpuDevice(id, info);
  device_storage_[id] = device;
}

void DeviceFactory::InsertGPUDevice(uint64_t id, DeviceInfo info) {
  Device *device = new GpuDevice(id, info);
  device_storage_[id] = device;
}

}

