#pragma once
#include "dag/dag.h"

namespace minerva {

class DagProcedure {
 public:
  virtual void Process(Dag&) = 0;
};

}
