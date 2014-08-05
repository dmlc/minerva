#pragma once
#include "op/physical.h"
#include "op/logical.h"
#include "common/nvector.h"
#include <string>
#include <vector>

namespace minerva {

struct VectorLoaderClosure {
  std::vector<float>* data;
};

class VectorLoaderOp: public LogicalDataGenFn, public PhysicalComputeFn, public ClosureTrait<VectorLoaderClosure> {
 public:
  void Execute(DataList&, DataList&, IMPL_TYPE);
  NVector<Chunk> Expand(const NVector<Scale>&);
  std::string Name() const;
};

}

