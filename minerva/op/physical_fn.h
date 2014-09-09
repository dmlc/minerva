#pragma once
#include "op/basic_fn.h"
#include "op/closure_trait.h"
#include "op/physical.h"
#include "op/device_info_trait.h"

namespace minerva {

class DataShard {
 public:
  DataShard(const PhysicalData&);
  DataShard(float* data, Scale size, Scale offset);
  // return data untransformed (NO memory copy)
  float* GetCpuData();
  float* GetGpuData();
  // return data transformed (may incur memory copy !!!)
  //float* GetTransformedCpuData();
  //float* GetTransformedGpuData();
  // Getters
  Scale Size();
  Scale Offset();
 private:
  float* data_;
  Scale size_, offset_;
};

typedef std::vector<DataShard> DataList;

class PhysicalComputeFn: public BasicFn, public virtual DeviceInfoTrait {
 public:
  virtual void Execute(DataList&, DataList&, ImplType) = 0;
};

template<class Closure>
class PhyComputeFnWithClosure: public PhysicalComputeFn, public ClosureTrait<Closure> {
 public:
  void Execute(DataList& inputs, DataList& outputs, ImplType impl_type) {
    FnBundle<Closure>::Call(inputs, outputs, ClosureTrait<Closure>::closure, impl_type);
  }
};

}
