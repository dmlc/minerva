#include "procedures/runtime_info_map.h"
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

RuntimeInfoMap::RuntimeInfoMap() {
}

void RuntimeInfoMap::AddNode(uint64_t id) {
  CHECK(state_sets_[static_cast<int>(NodeState::kBirth)].insert(id).second);
  CHECK(info_.emplace(id, RuntimeInfo{0, 0, NodeState::kBirth}).second);
}

void RuntimeInfoMap::RemoveNode(uint64_t id) {
  CHECK_EQ(state_sets_[static_cast<int>(info_.at(id).state)].erase(id), 1);
  CHECK_EQ(info_.erase(id), 1);
}

RuntimeInfo& RuntimeInfoMap::At(uint64_t id) {
  return info_.at(id);
}

NodeState RuntimeInfoMap::GetState(uint64_t id) {
  return info_.at(id).state;
}

void RuntimeInfoMap::ChangeState(uint64_t id, NodeState to) {
  NodeState old = info_.at(id).state;
  if (old != to) {
    info_.at(id).state = to;
    CHECK_EQ(state_sets_[static_cast<int>(old)].erase(id), 1);
    CHECK_EQ(state_sets_[static_cast<int>(to)].insert(id), 1);
  }
}

const unordered_set<uint64_t>& RuntimeInfoMap::GetNodesOfState(NodeState s) const {
  return state_sets_[static_cast<int>(s)];
}

}  // namespace minerva

