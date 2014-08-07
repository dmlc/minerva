#pragma once
#include "op/physical_fn.h"
#include "op/logical_fn.h"

namespace minerva {

template<class Closure>
class SharedDataGenFnWithClosure: public LogicalDataGenFn, public PhysicalComputeFn, public ClosureTrait<Closure> {
 public:
  void Execute(DataList&, DataList& outputs, IMPL_TYPE impl_type) {
    FnBundle<Closure>::Call(outputs, ClosureTrait<Closure>::closure, impl_type);
  }
};

template<class Closure>
class SharedComputeFnWithClosure: public LogicalComputeFn, public PhyComputeFnWithClosure<Closure> {
};

}

