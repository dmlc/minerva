#pragma once
#include "common/scale.h"
#include "common/nvector.h"
#include "chunk/chunk.h"
#include "op/logical_fn.h"
#include "context.h"

namespace minerva {

struct LogicalData {
  Scale size;
  LogicalDataGenFn* data_gen_fn;
  NVector<PartInfo> partitions;
  int extern_rc;
  LogicalData(): data_gen_fn(NULL), extern_rc(0) {}
  LogicalData(const Scale& s, LogicalDataGenFn* fn = NULL):
    size(s), data_gen_fn(fn), extern_rc(0) {}
  //DataNodeContext context; // TODO how to set context ?
};

struct LogicalOp {
  LogicalComputeFn* compute_fn;
};

}// end of namespace minerva
