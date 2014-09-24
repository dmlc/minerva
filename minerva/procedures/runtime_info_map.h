#pragma once
#include <cstdint>
#include <unordered_set>
#include <unordered_map>
#include <mutex>
#include <atomic>
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

struct RuntimeInfo {
  RuntimeInfo();
  std::atomic<int> num_triggers_needed;
  std::atomic<int> reference_count;
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
  void KillNode(uint64_t);
  std::unordered_set<uint64_t> dead_nodes();

 private:
  std::unordered_set<uint64_t> dead_nodes_;
  std::unordered_map<uint64_t, RuntimeInfo> info_;
  DISALLOW_COPY_AND_ASSIGN(RuntimeInfoMap);
};

}  // namespace minerva

