#pragma once
#include "common/scale.h"
#include "impl/impl.h"
#include "context.h"

namespace minerva {

class PhysicalComputeFn;

struct PhysicalData {
  PhysicalData(const Scale& s, uint64_t d, uint64_t id) : size(s), device_id(d), data_id(id) {
  }
  Scale size;
  int extern_rc = 0;
  uint64_t device_id;
  uint64_t data_id;
};

struct PhysicalOp {
  PhysicalComputeFn* compute_fn;
};

} // end of namespace minerva

