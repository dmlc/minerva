#pragma once
#include "op/basic_fn.h"
#include "op/closure_trait.h"
#include "op/physical.h"
#include "op/device_id_trait.h"
#include "op/context.h"

namespace minerva {

class DataShard {
 public:
  DataShard(float* data, const Scale& size);
  // Getters
  Scale size();
  float* data();

 private:
  float* data_;
  Scale size_;
};

typedef std::vector<DataShard> DataList;

class PhysicalComputeFn : public BasicFn, public DeviceIdTrait {
 public:
  virtual void Execute(const DataList&, const DataList&, const Context&) = 0;
};

template<typename Closure>
class PhyComputeFnWithClosure : public PhysicalComputeFn, public ClosureTrait<Closure> {
 public:
  void Execute(const DataList& inputs, const DataList& outputs, const Context& context) {
    FnBundle<Closure>::Call(inputs, outputs, ClosureTrait<Closure>::closure, context);
  }
};

template<typename Closure>
class PhyDataGenFnWithClosure : public PhysicalComputeFn, public ClosureTrait<Closure> {
 public:
  void Execute(const DataList&, const DataList& outputs, const Context& context) {
    FnBundle<Closure>::Call(outputs, ClosureTrait<Closure>::closure, context);
  }
};

}

