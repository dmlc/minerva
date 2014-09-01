#include "device_factory.h"

using namespace std;

namespace minerva {

void DeviceFactory::Reset() {
  allocated_ = 0;
}

void DeviceFactory::PrintDevice(DeviceInfo device_info) {
  printf("device id: %d", device_info.id);
}

DeviceInfo DeviceFactory::DefaultInfo() {
  DeviceInfo result;
  result.id = 0;
  if (allocated_ == 0) {
    ++allocated_;
  }
  result.cpu_list.push_back("localhost");
  result.gpu_list.push_back(0);
  result.num_streams.push_back(1);
  return result;
}

DeviceInfo DeviceFactory::GpuDeviceInfo(int gid) {
  DeviceInfo result;
  result.id = allocated_++;
  result.gpu_list.push_back(gid);
  result.num_streams.push_back(1);
  return result;
}

DeviceInfo DeviceFactory::GpuDeviceInfo(int gid, int num_stream) {
  DeviceInfo result;
  result.id = allocated_++;
  result.gpu_list.push_back(gid);
  result.num_streams.push_back(num_stream);
  return result;
}

Device DeviceFactory::GetDevice(int id) {
  return device_storage_.get(id);
}

}

