#pragma once
#include "common/scale.h"
#include "impl/impl.h"
#include "op.h"
#include "context.h"

namespace minerva {

struct PhysicalData;
struct PhysicalOp;
class PhysicalComputeFn;
class DataShard;

/////////////////////////////////////////////
// Physical dag data structures
/////////////////////////////////////////////
struct PhysicalData {
  Scale size, offset, offset_index;
  uint64_t data_id;
};

struct PhysicalOp {
  Place place;
  IMPL_TYPE impl_type;
  PhysicalComputeFn* compute_fn;
};

/////////////////////////////////////////////
// Execute function structures
/////////////////////////////////////////////
/**
 * A wrapper class for physical data
 */
class DataShard {
 public:
  DataShard(const PhysicalData& );
  DataShard(const DataShard& other);
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

class PhysicalComputeFn : public BasicFn {
 public:
  virtual void Execute(DataList& inputs, DataList& outputs, IMPL_TYPE impl_type) = 0;
};

template<class C>
class PhyComputeFnWithClosure :
  public PhysicalComputeFn, public ClosureTrait<C> {
 public:
  void Execute(DataList& inputs, DataList& outputs, IMPL_TYPE impl_type) {
    FnBundle<C>::Call(inputs, outputs, ClosureTrait<C>::closure, impl_type);
  }
};

} // end of namespace minerva
