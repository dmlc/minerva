#pragma once
#include <cstdint>
#include <unordered_set>
#include <unordered_map>
#include <atomic>
#include "common/common.h"
#include "common/concurrent_unordered_map.h"

namespace minerva {

enum class NodeState {
  kReady,
  kCompleted
};

std::ostream& operator<<(std::ostream&, NodeState);

struct RuntimeInfo {
  RuntimeInfo();
  int num_triggers_needed;
  int reference_count;
  NodeState state;
};

class RuntimeInfoMap {
 public:
  RuntimeInfoMap() = default;
  DISALLOW_COPY_AND_ASSIGN(RuntimeInfoMap);
  void AddNode(uint64_t);
  void RemoveNode(uint64_t);
  RuntimeInfo& At(uint64_t);
  NodeState GetState(uint64_t);
  void KillNode(uint64_t);

 private:
  ConcurrentUnorderedMap<uint64_t, RuntimeInfo> info_;
};

}  // namespace minerva

