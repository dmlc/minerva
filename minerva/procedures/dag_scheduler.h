#pragma once
#include "procedures/dag_procedure.h"
#include "procedures/node_state_map.h"
#include "dag/dag.h"
#include "dag/physical_dag.h"
#include "common/common.h"
#include <unordered_set>
#include <mutex>
#include <condition_variable>

namespace minerva {

class DagScheduler : public DagProcedure<PhysicalDag>, public DagMonitor<PhysicalDag> {
 public:
  DagScheduler(PhysicalDag*);
  // Wait for evaluation to finish
  void WaitForFinish();
  void GCNodes();
  int CalcTotalReferenceCount(PhysicalDataNode*);
  void OnIncrExternRC(PhysicalDataNode*, int);
  // DAG procedure
  void Process(const std::vector<uint64_t>&);
  // DAG monitor
  void OnCreateNode(DagNode*);
  void OnDeleteNode(DagNode*);
  void OnCreateEdge(DagNode*, DagNode*);
  NodeStateMap& node_states();

 private:
  struct RuntimeInfo {
    int num_triggers_needed;
    int reference_count;
    std::mutex* mutex;
  };
  void FreeDataNodeRes(PhysicalDataNode*);
  void ProcessNode(DagNode*);
  std::unordered_set<uint64_t> FindStartFrontier(const std::vector<uint64_t>&);
  NodeStateMap node_states_;
  std::unordered_map<uint64_t, RuntimeInfo> rt_info_;
  int num_nodes_yet_to_finish_;
  // TODO std::mutex scheduler_busy_;
  std::mutex finish_mutex_;
  std::condition_variable finish_cond_;
  DISALLOW_COPY_AND_ASSIGN(DagScheduler);
};

}  // namespace minerva
