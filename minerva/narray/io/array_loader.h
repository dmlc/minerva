#pragma once
#include "op/physical.h"
#include "op/logical.h"
#include "common/nvector.h"
#include <string>
#include <vector>

namespace minerva {

struct ArrayLoaderClosure {
  // TODO should be std::shared_ptr<float>
  float* data;
  Scale size;
};

class ArrayLoaderOp:
  public LogicalDataGenFn,
  public PhysicalComputeFn,
  public ClosureTrait<ArrayLoaderClosure> {
 public:
  void Execute(DataList&, DataList&, ImplType);
  NVector<Chunk> Expand(const NVector<Scale>&);
  std::string Name() const;
};

}

