#pragma once
#include "dag/dag_node.h"
#include "procedures/dag_procedure.h"
#include "procedures/thread_pool.h"
#include "common/common.h"
#include "common/concurrent_blocking_queue.h"
#include <cstdint>
#include <map>
#include <vector>
#include <queue>
#include <atomic>
#include <mutex>
#include <condition_variable>

namespace minerva {

struct NodeState {
  enum State {
    kNoNeed, // No need to execute
    kReady, // Need to execute
    kTarget // Target node
  } state;
  size_t dependency_counter;
};

class DagEngine : public DagProcedure {
 public:
  DagEngine();
  ~DagEngine();
  void Process(Dag&, std::vector<uint64_t>&);

 private:
  DISALLOW_COPY_AND_ASSIGN(DagEngine);
  void ParseDagState(Dag&);
  void FindRootNodes(Dag&, std::vector<uint64_t>&);
  void AppendSubsequentNodes(DagNode*, ThreadPool*);
  std::map<uint64_t, NodeState> node_states_;
  std::mutex node_states_mutex_;
  std::queue<DagNode*> ready_to_execute_queue_;
  size_t unresolved_counter_;
  std::mutex unresolved_counter_mutex_;
  std::condition_variable execution_finished_;
  ThreadPool thread_pool_;
};

}

