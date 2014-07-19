#pragma once
#include "common/scale.h"
#include "context.h"
#include "op.h"
#include "physical.h"
#include <vector>

namespace minerva {

struct PhysicalData {
  Scale size, offset, chunk_index;
  uint64_t data_id;
  PhysicalDataGenFn* data_gen_fn;
  PhysicalData() {}
  PhysicalData(const Scale& size): size(size), data_id(0), data_gen_fn(0) {}
};

}

