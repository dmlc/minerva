#pragma once
#include "op/runner_wrapper.h"
#include "op/closure.h"

namespace minerva {

struct PhysicalOp {
  RunnerWrapper::ID runner_id;
  ClosureBase* closure;
};

}

