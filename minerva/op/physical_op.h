#pragma once
#include "physical.h"
#include "closure.h"
#include "shared_op.h"

namespace minerva {

class AssembleOp: public PhyComputeFnWithClosure<AssembleClosure> {
  std::string Name() const {
    return "assemble";
  }
};

}
