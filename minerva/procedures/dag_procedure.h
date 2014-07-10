#pragma once
#include <vector>
#include "dag/logical.h"
#include "dag/physical.h"

namespace minerva {

template<class DagType>
class DagProcedure {
 public:
  virtual void Process(DagType&, std::vector<uint64_t>&) = 0;
};

class LogicalDagProcedure : public DagProcedure<LogicalDag> { };
class PhysicalDagProcedure : public DagProcedure<PhysicalDag> { };

}
