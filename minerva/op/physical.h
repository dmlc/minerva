#pragma once
#include "common/scale.h"
#include "context.h"
#include "op.h"

namespace minerva {

struct PhysicalData;
struct PhysicalOp;
class PhysicalDataGenFn;
class PhysicalComputeFn;
class DataShard;

/////////////////////////////////////////////
// Physical dag data structures
/////////////////////////////////////////////
struct PhysicalData {
  Scale size, offset, offset_index;
  uint64_t data_id;
  PhysicalDataGenFn* data_gen_fn;
};

struct PhysicalOp {
  Place place;
  int impl_type;
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
  DataShard(PhysicalData& );
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
  PhysicalData& data_info_;
};

class PhysicalDataGenFn : public BasicFn {
 public:
  virtual void Execute(DataShard output, int impl_type) = 0;
};

class PhysicalComputeFn : public BasicFn {
 public:
  virtual void Execute(std::vector<DataShard> inputs, 
      std::vector<DataShard> outputs, int impl_type) = 0;
};

template<class T>
class BundleTrait {
 protected:
  T fn_bundle_;
};

template<class B, class C>
class PhyDataGenFnTempl :
  public PhysicalDataGenFn,
  public BundleTrait<B>, public ClosureTrait<C> {
 public:
  void Execute(DataShard output, int impl_type) {
    BundleTrait<B>::fn_bundle_[impl_type](output, ClosureTrait<C>::closure);
  }
};

template<class B, class C>
class PhyComputeFnTempl :
  public PhysicalComputeFn,
  public BundleTrait<B>, public ClosureTrait<C> {
 public:
  void Execute(std::vector<DataShard> inputs, 
      std::vector<DataShard> outputs, int impl_type) {
    BundleTrait<B>::fn_bundle_[impl_type](inputs, outputs, ClosureTrait<C>::closure);
  }
};

} // end of namespace minerva
