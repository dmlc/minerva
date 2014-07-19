#pragma once
#include "common/scale.h"
#include "context.h"
#include "op.h"
#include <vector>

namespace minerva {

class PhysicalDataGenFn : public BasicFn {
};

class PhysicalComputeFn : public BasicFn {
 public:
  //virtual void Execute(std::vector<PhysicalData> inputs,
      //std::vector<PhysicalData> outputs, PhysicalOp& op) = 0;
};

} // end of namespace minerva

