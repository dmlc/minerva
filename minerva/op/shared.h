#pragma once

#include "logical.h"
#include "physical.h"

namespace minerva {

class SharedDataGenFn : public LogicalDataGenFn, public PhysicalDataGenFn {
};

class SharedComputeFn : public LogicalComputeFn, public PhysicalComputeFn {
};

} // end of namespace minerva
