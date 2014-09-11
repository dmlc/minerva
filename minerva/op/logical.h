#pragma once
#include "common/scale.h"
#include "common/nvector.h"
#include "chunk/chunk.h"
#include "op/logical_fn.h"
#include "context.h"

namespace minerva {

struct LogicalData {
  LogicalData(): data_gen_fn(NULL), extern_rc(0) {}
  LogicalData(const Scale& s, LogicalDataGenFn* fn = 0):
    size(s), data_gen_fn(fn), extern_rc(0) {}
  Scale size;
  LogicalDataGenFn* data_gen_fn;
  NVector<Scale> partitions;
  int extern_rc;
  uint64_t device_id;
};

struct LogicalOp {
  LogicalComputeFn* compute_fn;
};

}// end of namespace minerva
