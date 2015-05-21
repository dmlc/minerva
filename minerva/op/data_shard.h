#pragma once
#include <vector>
#include "common/scale.h"

namespace minerva {

struct DataShard {
  DataShard(float* data, Scale const& size) : data_(data), size_(size) {
  }
  float* const data_;
  Scale const& size_;
};

using DataList = std::vector<DataShard>;

}  // namespace minerva

