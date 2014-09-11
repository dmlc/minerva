#pragma once
#include "common/scale.h"
#include "impl/impl.h"
#include "context.h"

namespace minerva {

class PhysicalComputeFn;

struct PhysicalData {
  PhysicalData(): extern_rc(0), data_id(0), mapped_to_lnode(false), mapped_lnid(0) {
  }
  Scale size, offset, offset_index;
  int extern_rc;
  uint64_t data_id;
  uint64_t device_id;
  bool mapped_to_lnode;
  uint64_t mapped_lnid;
};

struct PhysicalOp {
  // TODO Use compute_fn->device_info to determine device
  ImplType impl_type;
  PhysicalComputeFn* compute_fn;
};

} // end of namespace minerva

