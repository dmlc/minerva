#include "procedures/node_state_map.h"
#include <glog/logging.h>

using namespace std;

namespace minerva {

ostream& operator<<(ostream& os, NodeState s) {
  switch (s) {
    case NodeState::kBirth:
      return os << "Birth";
    case NodeState::kReady:
      return os << "Ready";
    case NodeState::kCompleted:
      return os << "Completed";
    case NodeState::kDead:
      return os << "Dead";
    default:
      CHECK(false) << "unknown state";
      return os;
  }
}

void NodeStateMap::AddNode(uint64_t id, NodeState init_state) {
  states_[id] = init_state;
  state_sets_[static_cast<int>(init_state)].insert(id);
}

void NodeStateMap::RemoveNode(uint64_t id) {
  NodeState s = states_[id];
  states_.erase(id);
  state_sets_[static_cast<int>(s)].erase(id);
}

NodeState NodeStateMap::GetState(uint64_t id) const {
  return states_.find(id)->second;
}

void NodeStateMap::ChangeState(uint64_t id, NodeState to) {
  NodeState old = states_[id];
  if (old != to) {
    lock_guard<mutex> lck(mutex_);
    states_[id] = to;
    state_sets_[static_cast<int>(old)].erase(id);
    state_sets_[static_cast<int>(to)].insert(id);
  }
}

const unordered_set<uint64_t>& NodeStateMap::GetNodesOfState(NodeState s) const {
  return state_sets_[static_cast<int>(s)];
}

}  // namespace minerva

