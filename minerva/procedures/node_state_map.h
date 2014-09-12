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

std::ostream& operator<<(std::ostream&, NodeState);

const int kNumNodeStates = static_cast<int>(NodeState::kDead) + 1;

class NodeStateMap {
 public:
  void AddNode(uint64_t, NodeState);
  void RemoveNode(uint64_t);
  NodeState GetState(uint64_t) const;
  void ChangeState(uint64_t, NodeState);
  const std::unordered_set<uint64_t>& GetNodesOfState(NodeState) const;

 private:
  mutable std::mutex mutex_;
  std::unordered_map<uint64_t, NodeState> states_;
  std::unordered_set<uint64_t> state_sets_[kNumNodeStates];
  DISALLOW_COPY_AND_ASSIGN(NodeStateMap);
};

}  // namespace minerva

