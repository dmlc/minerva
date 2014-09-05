#include "op/physical_fn.h"
#include "system/minerva_system.h"
#include "system/data_store.h"

using namespace std;

namespace minerva {

DataShard::DataShard(const PhysicalData& d) {
  int device_id = d.device_info.id;
  data_ = MinervaSystem::Instance().GetDevice(device_id)->GetData(d.data_id);
  size_ = d.size;
}

DataShard::DataShard(float* data, Scale size): data_(data), size_(size) {
}

// return data untransformed (NO memory copy)
float* DataShard::GetCpuData() {
  return data_;
}

float* DataShard::GetGpuData() {
  return data_;
}

Scale DataShard::Offset() {
  return size_;
}

Scale DataShard::Size() {
  return size_;
}

}
