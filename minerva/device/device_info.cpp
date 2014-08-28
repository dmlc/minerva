#include <stdio.h>
#include "device_info.h"

using namespace std;

namespace minerva {

void DeviceFactory::Initialize() {
  allocated = 0;
}

void DeviceFactory::print_device(DeviceInfo device_info) {
  printf("device id: %d", device_info.id);
}

DeviceInfo DeviceFactory::default_info() {
  DeviceInfo result;
  result.id = allocated ++;
  result.CPUList.push_back("localhost");
  result.GPUList.push_back(0);
  result.numStreams.push_back(1);
  return result;
}

DeviceInfo DeviceFactory::gpu_device_info(int gid) {
  DeviceInfo result;
  result.id = allocated ++;
  result.GPUList.push_back(gid);
  result.numStreams.push_back(1);
  return result;
}

DeviceInfo DeviceFactory::gpu_device_info(int gid, int numStream) {
  DeviceInfo result;
  result.id = allocated ++;
  result.GPUList.push_back(gid);
  result.numStreams.push_back(numStream);
  return result;
}

}
