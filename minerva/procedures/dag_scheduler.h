#pragma once
#include "procedures/dag_procedure.h"
#include "dag/dag.h"
#include "dag/physical_dag.h"
#include "common/common.h"

namespace minerva {

class DagScheduler : public DagProcedure<PhysicalDag>, public DagMonitor<PhysicalDag> {
 public:
  DagScheduler();
  void Process();
 private:
  DISALLOW_COPY_AND_ASSIGN(DagScheduler);
};

}
