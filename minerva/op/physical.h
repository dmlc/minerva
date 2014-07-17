#pragma once
#include "common/scale.h"
#include "context.h"
#include "op.h"
#include "procedures/physical_engine.h"
#include <vector>

namespace minerva {

class PhysicalDataGenFn : public BasicFn {
};

class PhysicalComputeFn : public BasicFn {
 public:
  //virtual void Execute(std::vector<PhysicalData> inputs,
      //std::vector<PhysicalData> outputs, PhysicalOp& op) = 0;
};

struct PhysicalData {
  Scale size, offset, chunk_index;
  //DataNodeContext context; // TODO how to set context ?
  uint64_t data_id;
  PhysicalDataGenFn* data_gen_fn;
  PhysicalData() {}
  PhysicalData(const Scale& size): size(size), data_id(0), data_gen_fn(NULL) {}
};

struct PhysicalOp {
  //OpNodeContext context; // TODO how to set context ?
  //OpExecutor* executor; // TODO [jermaine] I think we don't need to set the function pointer here, because there might be several types of implementation for a single function which needs to be determined later.
  PhysicalEngine::RunnerID runner_id;
};

} // end of namespace minerva

