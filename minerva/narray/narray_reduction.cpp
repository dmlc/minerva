#include "narray.h"
#include "op/shared_op.h"
#include <iostream>

using namespace std;

namespace minerva {

// Lazy reductions
NArray NArray::Sum(int dim) {
  return Sum(Scale{dim});
}

NArray NArray::Sum(const Scale& dims) {
  auto size = Size();
  for (auto i: dims) {
    size[i] = 1;
  }
  ReductionOp* reduction_op = new ReductionOp;
  reduction_op->closure.type = SUM;
  reduction_op->closure.dims_to_reduce = dims;
  return NArray::Compute({*this}, {size}, reduction_op)[0];
}

NArray NArray::Max(int dim) {
  return Max(Scale{dim});
}

NArray NArray::Max(const Scale& dims) {
  auto size = Size();
  for (auto i: dims) {
    size[i] = 1;
  }
  ReductionOp* reduction_op = new ReductionOp;
  reduction_op->closure.type = MAX;
  reduction_op->closure.dims_to_reduce = dims;
  return NArray::Compute({*this}, {size}, reduction_op)[0];
}

NArray NArray::MaxIndex(int dim) {
  return MaxIndex(Scale(dim));
}
NArray NArray::MaxIndex(const Scale& dims) {
  // TODO
  return NArray();
}
// Non-lazy reduction
float NArray::Sum() {
  // TODO
  return 0;
}
float NArray::Max() {
  // TODO
  return 0;
}
int NArray::CountZero() {
  // TODO
  return 0;
}

} // end of namespace minerva

