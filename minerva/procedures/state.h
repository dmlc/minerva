#pragma once

#include <cstdint>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <iostream>
#include "dag/dag.h"

namespace minerva {

enum class NodeState {
  kBirth = 0,
  kReady,
  kCompleted,
  kDead,
};
const int kNumNodeStates = (int)NodeState::kDead + 1;

inline std::ostream& operator << (std::ostream& os, NodeState s) {
  switch(s) {
    case NodeState::kBirth:
      return os << "Birth";
    case NodeState::kReady:
      return os << "Ready";
    case NodeState::kCompleted:
      return os << "Completed";
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
    //state_sets_[NodeState::kBirth] = std::unordered_set<uint64_t>();
    //state_sets_[NodeState::kReady] = std::unordered_set<uint64_t>();
    //state_sets_[NodeState::kCompleted] = std::unordered_set<uint64_t>();
    //state_sets_[NodeState::kDead] = std::unordered_set<uint64_t>();
  }
  void OnCreateNode(DagNode* n) {
    AddNode(n->node_id(), NodeState::kBirth);
  }
  void OnDeleteNode(DagNode* n) {
    RemoveNode(n->node_id());
  }
  NodeState GetState(uint64_t id) const {
    std::lock_guard<std::mutex> lck(mutex_);
    return states_.find(id)->second;
  }
  void ChangeState(uint64_t id, NodeState to) {
    std::lock_guard<std::mutex> lck(mutex_);
    NodeState old = states_[id];
    if(old != to) {
      states_[id] = to;
      state_sets_[(int)old].erase(id);
      state_sets_[(int)to].insert(id);
    }
  }
  const std::unordered_set<uint64_t>& GetNodesOfState(NodeState s) const {
    return state_sets_[(int)s];
  }
 private:
  void AddNode(uint64_t id, NodeState init_state) {
    states_[id] = init_state;
    state_sets_[(int)init_state].insert(id);
  }
  void RemoveNode(uint64_t id) {
    NodeState s = states_[id];
    states_.erase(id);
    state_sets_[(int)s].erase(id);
  }
 private:
  mutable std::mutex mutex_;
  std::unordered_map<uint64_t, NodeState> states_;
  std::unordered_set<uint64_t> state_sets_[kNumNodeStates];
};

}
