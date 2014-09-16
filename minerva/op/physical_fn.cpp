#include "op/physical_fn.h"
#include "system/minerva_system.h"
#include "system/data_store.h"

using namespace std;

namespace minerva {

DataShard::DataShard(float* data, const Scale& size): data_(data), size_(size) {
}

Scale DataShard::size() {
  return size_;
}

float* DataShard::data() {
  return data_;
}

}  // namespace minerva

