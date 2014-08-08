#pragma once
#include <functional>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <condition_variable>
#include <thread>
#include <mutex>

#include "procedures/dag_procedure.h"
#include "procedures/state.h"
#include "common/common.h"
#include "common/thread_pool.h"

namespace minerva {

/*
 * PhysicalEngine makes following assumptions:
 * 1. Each PhysicalDataNode has one or zero predecessor.
 * 2. User is only allowed to add new nodes to the dag.
 * 3. PhysicalEngine is responsible for GC.
 */

/*struct NodeState {
  enum State {
    kNoNeed, // No need to execute
    kReady, // Need to execute
    kComplete
  } state;
};*/

class PhysicalEngine: public PhysicalDagProcedure, public PhysicalDagMonitor {
 public:
  PhysicalEngine();
  ~PhysicalEngine();
  void Process(PhysicalDag&, NodeStateMap<PhysicalDag>&,
      const std::vector<uint64_t>&);
  void OnCreateNode(DagNode* node);
  void OnDeleteNode(DagNode* node);
  void GCNodes(PhysicalDag&, NodeStateMap<PhysicalDag>& );

 private:
  struct RuntimeState {
    int dependency_counter;
    std::condition_variable* on_complete;
  };

 private:
  DISALLOW_COPY_AND_ASSIGN(PhysicalEngine);
  void Init();
  std::unordered_set<DagNode*> FindRootNodes(PhysicalDag& dag,
      NodeStateMap<PhysicalDag>&, const std::vector<uint64_t>&);
  void NodeRunner(DagNode*, NodeStateMap<PhysicalDag>& );
  void AppendTask(DagNode*, NodeStateMap<PhysicalDag>& );

  std::mutex node_states_mutex_;
  std::unordered_map<uint64_t, RuntimeState> rt_states_;

  ThreadPool thread_pool_;
};

}
