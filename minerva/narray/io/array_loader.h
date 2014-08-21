#pragma once
#include "op/physical.h"
#include "op/logical.h"
#include "common/nvector.h"
#include <string>
#include <vector>

namespace minerva {

struct ArrayLoaderClosure {
  std::shared_ptr<float> data;
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

