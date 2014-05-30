#pragma once
#include "procedures/dag_procedure.h"
#include "common/common.h"
#include <cstdint>
#include <map>

namespace minerva {

struct NodeState {
  enum State {
    kNoNeed,
    kReady
  } state;
};

class DagEngine : public DagProcedure {
 public:
  void Process(Dag&);

 private:
  DISALLOW_COPY_AND_ASSIGN(DagEngine);
  void ParseDagState(Dag&);
  std::map<uint64_t, NodeState> node_states_;
  // TODO private members including but not limited to
  // 1. Threadpool
  // 2. Execution state (like counter)
};

}
