#pragma once
#include "common/scale.h"
#include "op/closure.h"
#include "op/runner_wrapper.h"
#include <vector>

namespace minerva {

struct PhysicalData {
  PhysicalData() {}
  PhysicalData(const Scale& size): size(size), data_id(0), runner_id(-1) {}
  Scale size, offset, chunk_index;
  uint64_t data_id;
  RunnerWrapper::ID runner_id;
  Closure* closure;
};

}

