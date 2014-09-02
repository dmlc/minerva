#include "narray/io/array_loader.h"
#include "op/impl/basic.h"
#include "op/context.h"
#include <glog/logging.h>

using namespace std;

namespace minerva {

void ArrayLoaderOp::Execute(DataList& inputs, DataList& outputs, const Context& context) {
  CHECK_EQ(context.impl_type, ImplType::kBasic) << "vector loader only has basic implementation";
  Scale dst_start = Scale::Origin(closure.size.NumDims());
  for (auto& ds: outputs) {
    basic::NCopy(closure.data.get(), closure.size, ds.Offset(), ds.GetCpuData(), ds.Size(), dst_start, ds.Size());
  }
}

NVector<Chunk> ArrayLoaderOp::Expand(const NVector<Scale>& part_sizes) {
  ArrayLoaderOp* op = new ArrayLoaderOp;
  op->closure = closure;
  return {Chunk::Compute({}, part_sizes.ToVector(), op), part_sizes.Size()};
}

std::string ArrayLoaderOp::Name() const {
  return "array loader";
}

}

