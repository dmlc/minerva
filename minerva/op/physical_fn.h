#pragma once
#include "op/basic_fn.h"
#include "op/closure_trait.h"
#include "op/physical.h"

namespace minerva {

class DataShard {
 public:
  DataShard(const PhysicalData&);
  DataShard(const DataShard&);
  // return data untransformed (NO memory copy)
  float* GetCpuData();
  float* GetGpuData();
  // return data transformed (may incur memory copy !!!)
  float* GetTransformedCpuData();
  float* GetTransformedGpuData();
  // Getters
  Scale Size() const { return data_info_.size; }
  Scale Offset() const { return data_info_.offset; }
  Scale OffsetIndex() const { return data_info_.offset_index; }
 private:
  const PhysicalData& data_info_;
};

typedef std::vector<DataShard> DataList;

class PhysicalComputeFn: public BasicFn {
 public:
  virtual void Execute(DataList&, DataList&, IMPL_TYPE) = 0;
};

template<class Closure>
class PhyComputeFnWithClosure: public PhysicalComputeFn, public ClosureTrait<Closure> {
 public:
  void Execute(DataList& inputs, DataList& outputs, IMPL_TYPE impl_type) {
    FnBundle<Closure>::Call(inputs, outputs, ClosureTrait<Closure>::closure, impl_type);
  }
};

}

