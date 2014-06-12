#pragma once
#include "dag/dag.h"
#include <vector>

namespace minerva {

class DagProcedure {
 public:
  virtual void Process(Dag&, std::vector<uint64_t>&) = 0;
};

}
