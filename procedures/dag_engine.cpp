#include "procedures/dag_engine.h"
#include "dag/dag.h"
#include "dag/dag_node.h"
#include <map>

using namespace std; 

namespace minerva {

// void DagEngine::Process(Dag& dag, vector<uint64_t> targets) {
//   // TODO
//   // 1. Create meta info
//   // 2. Colorize
//   ParseDagState(dag);
//   map<uint64_t, NodeState>::iterator it;
//   for (auto i: targets) {
//     it = node_states_.find(i);
//     if (it == node_states_.end()) { // Node not found
//       continue;
//     }
//     it->second.state = kReady;
//     if (dag.GetNode(i)) {
//     }
//   }
// }

void DagEngine::ParseDagState(Dag& dag) {
  // Create NodeState for new nodes. Note that only new nodes are inserted.
  for (auto i: dag.index_to_node_) {
    NodeState n;
    n.state = NodeState::kNoNeed;
    node_states_.insert(make_pair(i.first, n));
  }
}

}
