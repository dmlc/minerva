#pragma once
#include "common/scale.h"
#include "op/closure.h"
#include "op/runner_wrapper.h"
#include <vector>

namespace minerva {

struct PhysicalData {
  PhysicalData() {}
  PhysicalData(const Scale& size): size(size) {}
  // TODO Ask storage to allocate a requested size
  Scale size, offset, chunk_index;
  uint64_t data_id = 0;
  RunnerWrapper::ID generator_id = 0;
  Closure* closure;
};

}

