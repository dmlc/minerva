#pragma once
#include "dag_procedure.h"
#include "common/thread_pool.h"
//#include "state.h"
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <queue>
#include <glog/logging.h>

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
class NodeStateMap {
 public:
  NodeStateMap() { }
  void AddNode(uint64_t id, NodeState init_state) {
    states_[id] = init_state;
    state_sets_[(int)init_state].insert(id);
  }
  void RemoveNode(uint64_t id) {
    NodeState s = states_[id];
    states_.erase(id);
    state_sets_[(int)s].erase(id);
  }
  NodeState GetState(uint64_t id) const {
    return states_.find(id)->second;
  }
  void ChangeState(uint64_t id, NodeState to) {
    NodeState old = states_[id];
    if(old != to) {
      states_[id] = to;
      std::lock_guard<std::mutex> lck(mutex_);
      state_sets_[(int)old].erase(id);
      state_sets_[(int)to].insert(id);
    }
  }
  const std::unordered_set<uint64_t>& GetNodesOfState(NodeState s) const {
    return state_sets_[(int)s];
  }
 private:
  mutable std::mutex mutex_;
  std::unordered_map<uint64_t, NodeState> states_;
  std::unordered_set<uint64_t> state_sets_[kNumNodeStates];
};

template<class DagType>
class DagEngine : public DagProcedure<DagType>, public DagMonitor<DagType> {
 public:
  DagEngine(ThreadPool& tp): thread_pool_(tp) {}
  void Process(DagType&, const std::vector<uint64_t>& nodes);
  void GCNodes(DagType& dag);
  int CalcTotalReferenceCount(typename DagType::DNode* );
  NodeStateMap& node_states() { return node_states_; }

  void OnCreateNode(DagNode* node);
  void OnDeleteNode(DagNode* node);
  void OnCreateEdge(DagNode* from, DagNode* to);
  void OnIncrExternRC(typename DagType::DNode*, int amount);

 protected:
  virtual void CreateNodeState(DagNode* ) {}
  virtual void DeleteNodeState(DagNode* ) {}
  virtual void SetUpReadyNodeState(DagNode* ) {}
  virtual void OnIncrInternalDep(typename DagType::DNode*, int amount) {}
  virtual void OnIncrExternalDep(typename DagType::DNode*, int amount) {}
  virtual void FreeDataNodeRes(typename DagType::DNode* ) {}
  virtual void ProcessNode(DagNode* node) = 0;

 protected:
  struct RuntimeInfo {
    int num_triggers_needed;
    int reference_count;
    std::mutex* mutex;
    std::condition_variable* on_complete;
  };
  void BottomUpScan(DagType& dag, const std::vector<uint64_t>& targets);
  void TopDownScan(DagType& dag, const std::vector<uint64_t>& targets);
  void AppendTask(DagNode* );
  void NodeTask(DagNode* );
  void TriggerSuccessors(DagNode* );

 protected:
  ThreadPool& thread_pool_;

  NodeStateMap node_states_;
  std::unordered_map<uint64_t, RuntimeInfo> rt_info_;
  std::unordered_set<uint64_t> start_frontier_;
};

template<class DagType>
void DagEngine<DagType>::Process(DagType& dag, const std::vector<uint64_t>& targets) {
  BottomUpScan(dag, targets);
  for(uint64_t ready_nid : node_states_.GetNodesOfState(NodeState::kReady)) {
    DagNode* ready_node = dag.GetNode(ready_nid);
    RuntimeInfo& ri = rt_info_[ready_nid];
    ri.num_triggers_needed = ready_node->predecessors_.size();
    if(ready_node->Type() == DagNode::OP_NODE) {
      for(DagNode* ready_op_succ : ready_node->successors_) {
        typename DagType::DNode* succ_dnode = dynamic_cast<typename DagType::DNode*>(ready_op_succ);
        if(CalcTotalReferenceCount(succ_dnode) == 0) {
          // very rare case: A data is generated completely useless.
          node_states_.ChangeState(ready_op_succ->node_id(), NodeState::kDead);
        } else {
          // mark successors of op node as kReady
          node_states_.ChangeState(ready_op_succ->node_id(), NodeState::kReady);
        }
      }
      ri.reference_count = -1;
    } else {
      typename DagType::DNode* ready_dnode = dynamic_cast<typename DagType::DNode*>(ready_node);
      ri.reference_count = CalcTotalReferenceCount(ready_dnode);
    }
    SetUpReadyNodeState(ready_node);
  }
  /*for(uint64_t start_nid : start_frontier_) {
    DagNode* start_node = dag.GetNode(start_nid);
    if(start_node->Type() == DagNode::DATA_NODE) {
      ResetDataNodeState(dag.GetDataNode(start_nid));
    }
  }*/
  TopDownScan(dag, targets);
}

template<class DagType>
void DagEngine<DagType>::OnCreateNode(DagNode* node) {
  node_states_.AddNode(node->node_id(), NodeState::kBirth);
  rt_info_[node->node_id()] = RuntimeInfo{0, 0, new std::mutex, nullptr};
  CreateNodeState(node);
}

template<class DagType>
void DagEngine<DagType>::OnDeleteNode(DagNode* node) {
  node_states_.RemoveNode(node->node_id());
  DeleteNodeState(node);
}
  
template<class DagType>
void DagEngine<DagType>::OnCreateEdge(DagNode* from, DagNode* to) {
  if(from->Type() == DagNode::DATA_NODE &&
      node_states_.GetState(from->node_id()) == NodeState::kCompleted) {
    typename DagType::DNode* dnode = dynamic_cast<typename DagType::DNode*>(from);
    OnIncrInternalDep(dnode, 1);
    rt_info_[from->node_id()].reference_count += 1;
  }
}

template<class DagType>
void DagEngine<DagType>::OnIncrExternRC(typename DagType::DNode* dnode, int amount) {
  if(node_states_.GetState(dnode->node_id()) == NodeState::kCompleted) {
    RuntimeInfo& ri = rt_info_[dnode->node_id()];
    ri.reference_count += amount;
    OnIncrExternalDep(dnode, amount);
    if(ri.reference_count == 0) {
      FreeDataNodeRes(dnode);
      node_states_.ChangeState(dnode->node_id(), NodeState::kDead);
    }
  }
}
  
template<class DagType>
void DagEngine<DagType>::BottomUpScan(DagType& dag, const std::vector<uint64_t>& targets) {
  start_frontier_.clear();
  std::queue<uint64_t> queue;
  for(uint64_t tgtid : targets) {
    if(node_states_.GetState(tgtid) != NodeState::kCompleted) {
      queue.push(tgtid);
    }
  }
  while(!queue.empty()) {
    uint64_t nid = queue.front();
    DagNode* node = dag.GetNode(nid);
    queue.pop();
    node_states_.ChangeState(nid, NodeState::kReady);
    int pred_count = 0;
    for(DagNode* pred : node->predecessors_) {
      NodeState pred_state = node_states_.GetState(pred->node_id());
      switch(pred_state) {
        case NodeState::kBirth:
          queue.push(pred->node_id());
          ++pred_count;
          break;
        case NodeState::kReady:
          ++pred_count;
          break;
        case NodeState::kCompleted:
          break;
        case NodeState::kDead:
        default:
          CHECK(false) << "invalid node state (" << pred_state << ") on dependency path";
          break;
      }
    }
    if(pred_count == 0) {
      start_frontier_.insert(nid);
    }
  }
}
  
template<class DagType>
void DagEngine<DagType>::TopDownScan(DagType& dag, const std::vector<uint64_t>& targets) {
  for(uint64_t start_nid : start_frontier_) {
    AppendTask(dag.GetNode(start_nid));
  }
  // Waiting execution to complete
  for(uint64_t tgtid : targets) {
    std::unique_lock<std::mutex> lock(*rt_info_[tgtid].mutex);
    LOG(INFO) << "Wait for node (id=" << tgtid << ") finish.";
    if (node_states_.GetState(tgtid) != NodeState::kCompleted) {
      RuntimeInfo& ri = rt_info_[tgtid];
      ri.on_complete = new std::condition_variable;
      ri.on_complete->wait(lock);
      delete ri.on_complete;
      ri.on_complete = nullptr;
    }
    LOG(INFO) << "Node (id=" << tgtid << ") complete.";
  }
  DLOG(INFO) << "Wait for thread to be quiet";
  thread_pool_.WaitForAllFinished();
}

template<class DagType>
void DagEngine<DagType>::AppendTask(DagNode* node) {
  thread_pool_.Push(std::bind(&DagEngine<DagType>::NodeTask, this, node));
}
  
template<class DagType>
int DagEngine<DagType>::CalcTotalReferenceCount(typename DagType::DNode* dnode) {
  int count = dnode->data_.extern_rc;
  for(DagNode* succ : dnode->successors_) {
    NodeState succ_state = node_states_.GetState(succ->node_id());
    if(succ_state == NodeState::kBirth || succ_state == NodeState::kReady) {
      ++count;
    }
  }
  return count;
}

template<class DagType>
void DagEngine<DagType>::NodeTask(DagNode* node) {
  uint64_t nid = node->node_id();
  DLOG(INFO) << "Process node#" << nid;
  CHECK_EQ(node_states_.GetState(nid), NodeState::kReady);
  ProcessNode(node); // ATTENTION: NO lock protected!
  // change state
  if(node->Type() == DagNode::OP_NODE) {
    node_states_.ChangeState(nid, NodeState::kDead); // the op node is executed thus could be GCed
    // check all its predecessors
    for(DagNode* pred : node->predecessors_) {
      RuntimeInfo& pred_ri = rt_info_[pred->node_id()];
      std::lock_guard<std::mutex> lck(*pred_ri.mutex);
      CHECK_GE(--pred_ri.reference_count, 0) << "invalid reference_count for node#" << pred->node_id();
      if(pred_ri.reference_count == 0) {
        FreeDataNodeRes(dynamic_cast<typename DagType::DNode*>(pred));
        node_states_.ChangeState(pred->node_id(), NodeState::kDead);
      }
    }
  } else {
    // data node is changed to Complete state
    node_states_.ChangeState(nid, NodeState::kCompleted);
  }

  TriggerSuccessors(node);

  RuntimeInfo& ri = rt_info_[node->node_id()];
  std::lock_guard<std::mutex> lck(*ri.mutex);
  if(ri.on_complete != nullptr) {
    DLOG(INFO) << "Notify node#" << node->node_id() << " is finished";
    ri.on_complete->notify_all();
  }
}
  
template<class DagType>
void DagEngine<DagType>::TriggerSuccessors(DagNode* node) {
  for(DagNode* succ : node->successors_) {
    RuntimeInfo& ri = rt_info_[succ->node_id()];
    std::lock_guard<std::mutex> lock(*ri.mutex);
    NodeState succ_state = node_states_.GetState(succ->node_id());
    CHECK_NE(succ_state, NodeState::kCompleted);
    CHECK_NE(succ_state, NodeState::kDead);
    if(succ_state == NodeState::kReady) {
      CHECK_GE(--ri.num_triggers_needed, 0) << "wrong #triggers for node#" << succ->node_id();
      if(ri.num_triggers_needed == 0)
        AppendTask(succ);
    }
  }
}
  
template<class DagType>
void DagEngine<DagType>::GCNodes(DagType& dag) {
  std::vector<uint64_t> dead_nodes; //survived_nodes;
  /*for(uint64_t pending_nid : node_states_.GetNodesOfState(NodeState::kPending)) {
    DagNode* pending_node = dag.GetNode(pending_nid);
    CHECK_EQ(pending_node->Type(), DagNode::DATA_NODE) << "only data node could be in Pending state";
    typename DagType::DNode* pending_dnode = dynamic_cast<typename DagType::DNode*>(pending_node);
    if(CalcTotalReferenceCount(pending_dnode) == 0) {
      dead_nodes.push_back(pending_nid);
    } else {
      survived_nodes.push_back(pending_nid);
    }
    ResetDataNodeState(pending_dnode);
  }*/
  for(uint64_t dead_nid : node_states_.GetNodesOfState(NodeState::kDead)) {
    dead_nodes.push_back(dead_nid);
  }
  for(uint64_t dead_nid : dead_nodes) {
    dag.DeleteNode(dead_nid);
  }
  /*for(uint64_t surv_nid : survived_nodes) {
    node_states_.ChangeState(surv_nid, NodeState::kCompleted);
  }*/
}

} // end of namespace minerva
