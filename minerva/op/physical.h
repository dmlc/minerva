#pragma once
#include "common/scale.h"
#include "impl/impl.h"
#include "context.h"

namespace minerva {

class PhysicalComputeFn;

struct PhysicalData {
  Scale size, offset, offset_index;
  int extern_rc;
  uint64_t data_id;
  PhysicalData(): extern_rc(0), data_id(0) {
  }
};

struct PhysicalOp {
  Place place;
  IMPL_TYPE impl_type;
  PhysicalComputeFn* compute_fn;
};

} // end of namespace minerva

