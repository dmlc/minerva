#include "narray/narray.h"
#include "op/physical_op.h"
#include <glog/logging.h>

using namespace std;

namespace minerva {

// Lazy reductions
NArray NArray::Sum(int dim) const {
  return Sum(Scale{dim});
}

NArray NArray::Sum(const Scale& dims) const {
  CHECK_GT(dims.NumDims(), 0) << "nothing to reduce";
  auto size = Size();
  for (auto i : dims) {
    CHECK(0 <= i && i < static_cast<int>(size.NumDims())) << "dim out of bound";
    size[i] = 1;
  }
  ReductionOp* reduction_op = new ReductionOp();
  reduction_op->closure.type = ReductionType::kSum;
  reduction_op->closure.dims_to_reduce = dims;
  return NArray::ComputeOne({*this}, size, reduction_op);
}

NArray NArray::Max(int dim) const {
  return Max(Scale{dim});
}

NArray NArray::Max(const Scale& dims) const {
  CHECK_GT(dims.NumDims(), 0) << "nothing to reduce";
  auto size = Size();
  for (auto i : dims) {
    CHECK(0 <= i && i < static_cast<int>(size.NumDims())) << "dim out of bound";
    size[i] = 1;
  }
  ReductionOp* reduction_op = new ReductionOp();
  reduction_op->closure.type = ReductionType::kMax;
  reduction_op->closure.dims_to_reduce = dims;
  return NArray::ComputeOne({*this}, size, reduction_op);
}

NArray NArray::MaxIndex(int dim) const {
  auto size = Size();
  CHECK(0 <= dim && dim < static_cast<int>(size.NumDims())) << "dim out of bound";
  size[dim] = 1;
  MaxIndexOp* op = new MaxIndexOp();
  op->closure.dim = dim;
  return NArray::ComputeOne({*this}, size, op);
}

// Non-lazy reductions
float NArray::Sum() const {
  // TODO
  CHECK(false) << "not implemented";
  return 0;
}

float NArray::Max() const {
  // TODO
  CHECK(false) << "not implemented";
  return 0;
}

int NArray::CountZero() const {
  auto ptr = Get();
  auto value = ptr.get();
  int size = Size().Prod();
  int counter = 0;
  for (int i = 0; i < size; ++i) {
    if (!value[i]) {
      ++counter;
    }
  }
  return counter;
}

} // end of namespace minerva

