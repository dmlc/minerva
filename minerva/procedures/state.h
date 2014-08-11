#pragma once

#include <cstdint>
#include <set>
#include <unordered_map>
#include <iostream>
#include "dag/dag.h"

namespace minerva {

enum class NodeState {
  kBirth = 0,
  kReady,
  kCompleted,
  kNeedGC,
  kDead,
  kNumStates,
};

inline std::ostream& operator << (std::ostream& os, NodeState s) {
  switch(s) {
    case NodeState::kBirth:
      return os << "Birth";
    case NodeState::kReady:
      return os << "Ready";
    case NodeState::kCompleted:
      return os << "Completed";
    case NodeState::kNeedGC:
      return os << "NeedGC";
    case NodeState::kDead:
      return os << "Dead";
    default:
      return os << "Unknown state";
  }
}

template<class DagType>
class NodeStateMap : public DagMonitor<DagType> {
 public:
  NodeStateMap() {
    state_sets_[NodeState::kBirth] = std::set<uint64_t>();
    state_sets_[NodeState::kReady] = std::set<uint64_t>();
    state_sets_[NodeState::kCompleted] = std::set<uint64_t>();
    state_sets_[NodeState::kNeedGC] = std::set<uint64_t>();
    state_sets_[NodeState::kDead] = std::set<uint64_t>();
  }
  void OnCreateNode(DagNode* n) {
    AddNode(n->node_id(), NodeState::kBirth);
  }
  void OnDeleteNode(DagNode* n) {
    RemoveNode(n->node_id());
  }
  NodeState GetState(uint64_t id) const {
    return states_.find(id)->second;
  }
  void ChangeState(uint64_t id, NodeState to) {
    NodeState old = states_[id];
    if(old != to) {
      states_[id] = to;
      state_sets_[old].erase(id);
      state_sets_[to].insert(id);
    }
  }
  const std::set<uint64_t>& GetNodesOfState(NodeState s) const {
    return state_sets_.find(s)->second;
  }
 private:
  void AddNode(uint64_t id, NodeState init_state) {
    states_[id] = init_state;
    state_sets_[init_state].insert(id);
  }
  void RemoveNode(uint64_t id) {
    NodeState s = states_[id];
    states_.erase(id);
    state_sets_[s].erase(id);
  }
 private:
  std::unordered_map<uint64_t, NodeState> states_;
  std::map<NodeState, std::set<uint64_t>> state_sets_;
};

}
