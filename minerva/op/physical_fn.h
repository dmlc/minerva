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
  DataShard(const DataShard&);
  // return data untransformed (NO memory copy)
  float* GetCpuData();
  float* GetGpuData();
  // return data transformed (may incur memory copy !!!)
  float* GetTransformedCpuData();
  float* GetTransformedGpuData();
  // Getters
  const Scale& Size() const { return data_info_.size; }
  const Scale& Offset() const { return data_info_.offset; }
  const Scale& OffsetIndex() const { return data_info_.offset_index; }
 private:
  const PhysicalData& data_info_;
};

typedef std::vector<DataShard> DataList;

class PhysicalComputeFn: public BasicFn, public virtual DeviceInfoTrait {
 public:
  virtual void Execute(const DataList&, const DataList&, const Context&) = 0;
};

template<typename Closure>
class PhyComputeFnWithClosure: public PhysicalComputeFn, public ClosureTrait<Closure> {
 public:
  void Execute(const DataList& inputs, const DataList& outputs, const Context& context) {
    FnBundle<Closure>::Call(inputs, outputs, ClosureTrait<Closure>::closure, context);
  }
};

template<typename Closure>
class PhyDataGenFnWithClosure : public PhysicalComputeFn, public ClosureTrait<Closure> {
 public:
  void Execute(const DataList& outputs, const Context& context) {
    FnBundle<Closure>::Call(outputs, ClosureTrait<Closure>::closure, context);
  }
};

}

