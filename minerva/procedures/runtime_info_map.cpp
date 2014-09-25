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

RuntimeInfo::RuntimeInfo() : num_triggers_needed(0), reference_count(0), state(NodeState::kBirth) {
}

RuntimeInfoMap::RuntimeInfoMap() {
}

void RuntimeInfoMap::AddNode(uint64_t id) {
  if (info_.find(id) != info_.end()) {
    CHECK(false);
  }
  info_[id];
}

void RuntimeInfoMap::RemoveNode(uint64_t id) {
  if (info_.at(id).state == NodeState::kDead) {
    CHECK_EQ(dead_nodes_.erase(id), 1);
  }
  CHECK_EQ(info_.erase(id), 1);
}

RuntimeInfo& RuntimeInfoMap::At(uint64_t id) {
  return info_.at(id);
}

NodeState RuntimeInfoMap::GetState(uint64_t id) {
  return info_.at(id).state;
}

void RuntimeInfoMap::KillNode(uint64_t id) {
  CHECK_EQ(info_.at(id).state, NodeState::kDead);
  CHECK(dead_nodes_.insert(id).second);
}

unordered_set<uint64_t> RuntimeInfoMap::dead_nodes() {
  return dead_nodes_;
}

}  // namespace minerva

