#pragma once
#include "common/scale.h"
#include "op/closure.h"
#include "op/runner_wrapper.h"
#include <vector>

namespace minerva {

struct PhysicalData {
  PhysicalData();
  PhysicalData(const Scale&);
  // TODO offset and chunk_index not used for now
  Scale size, offset, chunk_index;
  uint64_t data_id = 0;
  RunnerWrapper::ID generator_id = 0;
  ClosureBase* closure;
};

}

