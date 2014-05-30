#include "procedures/dag_engine.h"
#include "dag/dag.h"
#include "dag/dag_node.h"
#include <map>
#include <queue>

using namespace std; 

namespace minerva {

void DagEngine::Process(Dag& dag, vector<uint64_t>& targets) {
  ParseDagState(dag);
  FindRootNodes(dag, targets);
}

void DagEngine::ParseDagState(Dag& dag) {
  // Create NodeState for new nodes. Note that only new nodes are inserted.
  for (auto i: dag.index_to_node_) {
    NodeState n;
    n.state = NodeState::kNoNeed;
    node_states_.insert(make_pair(i.first, n));
  }
}

void DagEngine::FindRootNodes(Dag& dag, vector<uint64_t>& targets) {
  queue<uint64_t> ready_node_queue;
  for (auto i: targets) {
    auto it = node_states_.find(i);
    if (it == node_states_.end()) { // Node not found
      continue;
    }
    ready_node_queue.push(i);
  }
  while (!ready_node_queue.empty()) {
    uint64_t cur = ready_node_queue.front();
    ready_node_queue.pop();
    auto& it = node_states_[cur];
    auto node = dag.index_to_node_[cur];
    it.state = NodeState::kReady; // Set state to ready
    it.dependency_counter = node->predecessors_.size();
    if (node->predecessors_.empty()) {
      // Add to execution queue
      ready_to_execute_queue_.Push(node);
    } else {
      // Traverse predecessors
      for (auto i: node->predecessors_) {
        if (node_states_[i->node_id_].state == NodeState::kReady) { // Already visited
          continue;
        }
        ready_node_queue.push(i->node_id_);
      }
    }
  }
}

}
