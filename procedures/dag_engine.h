#pragma once
#include "procedures/dag_procedure.h"
#include "common/common.h"
#include "common/concurrent_blocking_queue.h"
#include <cstdint>
#include <map>
#include <vector>

namespace minerva {

struct NodeState {
  enum State {
    kNoNeed,
    kReady
  } state;
  size_t dependency_counter;
};

class DagEngine : public DagProcedure {
 public:
  void Process(Dag&, std::vector<uint64_t>&);

 private:
  DISALLOW_COPY_AND_ASSIGN(DagEngine);
  void ParseDagState(Dag&);
  void FindRootNodes(Dag&, std::vector<uint64_t>&);
  std::map<uint64_t, NodeState> node_states_;
  // TODO private members including but not limited to
  // 1. Threadpool
};

}
