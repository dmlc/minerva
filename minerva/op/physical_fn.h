#pragma once
#include "op/basic_fn.h"
#include "op/physical.h"
#include "op/context.h"
#include "op/impl/impl.h"
#include "common/common.h"

namespace minerva {

struct DataShard {
  DataShard(float* data, const Scale& size) : data_(data), size_(size) {
  }
  float *const data_;
  const Scale& size_;
};

typedef std::vector<DataShard> DataList;

class ComputeFn : public BasicFn {
 public:
  virtual void Execute(const DataList&, const DataList&, const Context&) = 0;
};

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
