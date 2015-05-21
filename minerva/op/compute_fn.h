#pragma once
#include "op/basic_fn.h"
#include "op/data_shard.h"

namespace minerva {

struct Context;

class ComputeFn : public BasicFn {
 public:
  virtual void Execute(DataList const&, DataList const&, Context const&) = 0;
};

}  // namespace minerva

