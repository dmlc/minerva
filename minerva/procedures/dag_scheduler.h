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
  void WaitForFinish();
  void GCNodes();
  int CalcTotalReferenceCount(PhysicalDataNode*);
  NodeStateMap& node_states();
  void OnIncrExternRC(PhysicalDataNode*, int);
  // DAG monitor
  void OnCreateNode(DagNode*);
  void OnDeleteNode(DagNode*);
  void OnCreateEdge(DagNode*, DagNode*);

 private:
  void FreeDataNodeRes(PhysicalDataNode*);
  void ProcessNode(DagNode*);
  std::unordered_set<uint64_t>
  DISALLOW_COPY_AND_ASSIGN(DagScheduler);
};

}
