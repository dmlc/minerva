#include "narray/io/array_loader.h"
#include "op/impl/basic.h"
#include "op/context.h"
#include <glog/logging.h>

using namespace std;

namespace minerva {

void ArrayLoaderOp::Execute(const DataList&, const DataList& outputs, const Context& context) {
  CHECK_EQ(context.impl_type, ImplType::kBasic) << "vector loader only has basic implementation";
  CHECK_EQ(outputs.size(), 1) << "takes 1 output";
  CHECK(closure.data) << "probably executed";
  memcpy(outputs[0].data(), closure.data.get(), closure.size.Prod() * sizeof(float));
  closure.data.reset();
}


std::string ArrayLoaderOp::Name() const {
  return "array loader";
}

}

