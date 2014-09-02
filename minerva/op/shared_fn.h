#pragma once
#include "op/physical_fn.h"
#include "op/logical_fn.h"
#include "op/context.h"

namespace minerva {

template<class Closure>
class SharedDataGenFnWithClosure: public LogicalDataGenFn, public PhysicalComputeFn, public ClosureTrait<Closure> {
 public:
  void Execute(DataList&, DataList& outputs, const Context& context) {
    FnBundle<Closure>::Call(outputs, ClosureTrait<Closure>::closure, context);
  }
};

template<class Closure>
class SharedComputeFnWithClosure: public LogicalComputeFn, public PhyComputeFnWithClosure<Closure> {
};

}

