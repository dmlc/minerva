#pragma once

#include <sstream>
#include "logical.h"
#include "closure.h"
#include "shared_op.h"

namespace minerva {

struct PartitionClosure {
  NVector<Scale> partitions;
};

class PartitionOp :
  public LogicalComputeFnTemp<1, 1>,
  public ClosureTrait<PartitionClosure> {
 public:
  NVector<Chunk> ExpandReal(const NVector<Chunk>& input) {
    // TODO
    return input;
  }
  std::string Name() const {
    return "re-part";
  }
};

} // end of namespace minerva
