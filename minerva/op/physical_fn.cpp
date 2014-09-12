#include "op/physical_fn.h"
#include "system/minerva_system.h"

namespace minerva {

DataShard::DataShard(const PhysicalData& d) {
  data_ = MinervaSystem::Instance().GetDevice(d.device_id)->GetData(d.data_id);
  size_ = d.size;
  offset_ = d.offset;
}

DataShard::DataShard(float* data, Scale size, Scale offset): data_(data), size_(size), offset_(offset) {
}

// return data untransformed (NO memory copy)
float* DataShard::GetCpuData() {
  return data_;
}

float* DataShard::GetGpuData() {
  return data_;
}

Scale DataShard::Offset() {
  return offset_;
}

Scale DataShard::Size() {
  return size_;
}

}
