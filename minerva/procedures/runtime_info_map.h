#pragma once
#include <cstdint>
#include <unordered_set>
#include <unordered_map>
#include <mutex>
#include <iostream>
#include "common/common.h"

namespace minerva {

enum class NodeState {
  kBirth = 0,
  kReady,
  kCompleted,
  kDead
};

const int kNumNodeStates = static_cast<int>(NodeState::kDead) + 1;

std::ostream& operator<<(std::ostream&, NodeState);

struct RuntimeInfo {
  int num_triggers_needed;
  int reference_count;
  // `state` should only be modified through `RuntimeInfoMap::ChangeState`
  NodeState state;
};

class RuntimeInfoMap {
 public:
  RuntimeInfoMap();
  void AddNode(uint64_t);
  void RemoveNode(uint64_t);
  RuntimeInfo& At(uint64_t);
  NodeState GetState(uint64_t);
  void ChangeState(uint64_t, NodeState);
  const std::unordered_set<uint64_t>& GetNodesOfState(NodeState) const;
  // Require manual locking
  std::recursive_mutex busy_mutex_;

 private:
  std::unordered_set<uint64_t> state_sets_[kNumNodeStates];
  std::unordered_map<uint64_t, RuntimeInfo> info_;
  DISALLOW_COPY_AND_ASSIGN(RuntimeInfoMap);
};

}  // namespace minerva

