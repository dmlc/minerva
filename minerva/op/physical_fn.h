#pragma once
#include "op/basic_fn.h"
#include "op/closure_trait.h"
#include "op/physical.h"
#include "op/device_info_trait.h"
#include "op/context.h"

namespace minerva {

class DataShard {
 public:
  DataShard(const PhysicalData&);
  DataShard(float* data, Scale size, Scale offset);
  // Getters
  Scale Size();
  float* data();

 private:
  float* data_;
  Scale size_, offset_;
};

typedef std::vector<DataShard> DataList;

class PhysicalComputeFn : public BasicFn, public DeviceInfoTrait {
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

