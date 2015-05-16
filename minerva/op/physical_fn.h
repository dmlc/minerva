#pragma once
#include "op/basic_fn.h"
#include "op/physical.h"
#include "op/impl/impl.h"
#include "common/common.h"
#include "op/data_shard.h"
#include "op/compute_fn.h"

namespace minerva {

struct Context;

template<typename Closure>
class ComputeFnWithClosure : public ComputeFn, public ClosureTrait<Closure> {
 public:
  void Execute(const DataList& inputs, const DataList& outputs, const Context& context) {
    FnBundle<Closure>::Call(inputs, outputs, ClosureTrait<Closure>::closure, context);
  }
};

template<typename Closure>
class PhyDataGenFnWithClosure : public ComputeFn, public ClosureTrait<Closure> {
 public:
  void Execute(const DataList&, const DataList& outputs, const Context& context) {
    FnBundle<Closure>::Call(outputs, ClosureTrait<Closure>::closure, context);
  }
};

}
