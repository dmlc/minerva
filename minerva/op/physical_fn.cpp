#include "op/physical_fn.h"
#include "system/minerva_system.h"
#include "system/data_store.h"

using namespace std;

namespace minerva {

DataShard::DataShard(const PhysicalData& d) {
  size_ = d.size;
}

DataShard::DataShard(float* data, Scale size, Scale offset): data_(data), size_(size), offset_(offset) {
}

Scale DataShard::Size() {
  return size_;
}

float* DataShard::data() {
  return data_;
}

}

