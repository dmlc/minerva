#pragma once
#include "common/scale.h"
#include "impl/impl.h"
#include "context.h"
#include "device/device_info.h"

namespace minerva {

class PhysicalComputeFn;

struct PhysicalData {
  PhysicalData() {
  }
  PhysicalData(const Scale& s, DeviceInfo info, uint64_t id): PhysicalData(s), device_info(info), data_id(id) {
  }
  Scale size;
  int extern_rc = 0;
  DeviceInfo device_info;
  uint64_t data_id = 0;
};

struct PhysicalOp {
  // TODO Use compute_fn->device_info to determine device
  ImplType impl_type;
  PhysicalComputeFn* compute_fn;
};

} // end of namespace minerva

