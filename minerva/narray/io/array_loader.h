#pragma once
#include "op/context.h"
#include "op/physical_fn.h"
#include <string>
#include <memory>
#include <vector>

namespace minerva {

struct ArrayLoaderClosure {
  std::shared_ptr<float> data;
  Scale size;
};

class ArrayLoaderOp :
  public PhysicalComputeFn,
  public ClosureTrait<ArrayLoaderClosure> {
 public:
  void Execute(const DataList&, const DataList&, const Context&);
  std::string Name() const;
};

}

