#include "narray/io/vector_loader.h"
#include <glog/logging.h>

using namespace std;

namespace minerva {

void VectorLoaderOp::Execute(DataList& inputs, DataList& outputs, IMPL_TYPE impl_type) {
  CHECK_EQ(impl_type, BASIC) << "vector loader only has basic implementation";
}

NVector<Chunk> VectorLoaderOp::Expand(const NVector<Scale>& part_sizes) {
  VectorLoaderOp* op = new VectorLoaderOp;
  op->closure = closure;
  return {Chunk::Compute({}, part_sizes.ToVector(), op), part_sizes.Size()};
}

std::string VectorLoaderOp::Name() const {
  return "vector loader";
}

}

