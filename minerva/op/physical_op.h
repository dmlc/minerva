#pragma once
#include "physical.h"
#include "closure.h"
#include "shared_op.h"

namespace minerva {

// TODO No need in having those since we are not dealing with partitioning any more
class AssembleOp: public PhyComputeFnWithClosure<AssembleClosure> {
  std::string Name() const {
    return "assemble";
  }
};

class SplitOp : public PhyComputeFnWithClosure<SplitClosure> {
  std::string Name() const {
    return "split";
  }
};

}
