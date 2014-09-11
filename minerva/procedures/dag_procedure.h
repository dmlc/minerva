#pragma once
#include <vector>
#include <glog/logging.h>
#include "dag/logical_dag.h"
#include "dag/physical_dag.h"

namespace minerva {

template<class DagType>
class DagProcedure {
 public:
  virtual void Process(DagType&, const std::vector<uint64_t>&) = 0;
};

}

