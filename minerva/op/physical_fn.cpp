#include "op/physical_fn.h"
#include "system/minerva_system.h"

using namespace std;

namespace minerva {

DataShard::DataShard(float* data, const Scale& size): data_(data), size_(size) {
}

const Scale& DataShard::size() const {
  return size_;
}

float* DataShard::data() const {
  return data_;
}

}  // namespace minerva

