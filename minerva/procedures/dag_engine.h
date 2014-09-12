#pragma once
#include "dag_procedure.h"
#include "common/thread_pool.h"
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <queue>
#include <glog/logging.h>

namespace minerva {

template<typename DagType>
class DagEngine : public DagProcedure<DagType>, public DagMonitor<DagType> {
 public:
  DagEngine(ThreadPool& tp): thread_pool_(tp) {}
  void Process(DagType& dag, const std::vector<uint64_t>& nodes);
  void WaitForFinish();
  void GCNodes(DagType& dag);
  int CalcTotalReferenceCount(typename DagType::DNode* node);
  NodeStateMap& node_states() { return node_states_; }
  virtual void OnIncrExternRC(typename DagType::DNode*, int amount);
  // DAG monitor
  void OnCreateNode(DagNode* node);
  void OnDeleteNode(DagNode* node);
  void OnCreateEdge(DagNode* from, DagNode* to);

 protected:
  virtual void FreeDataNodeRes(typename DagType::DNode* ) {}
  virtual void ProcessNode(DagNode* node) = 0;
  virtual std::unordered_set<uint64_t> FindStartFrontier(DagType& dag, const std::vector<uint64_t>& targets) = 0;
  virtual void PrepareProcess() { }
  virtual void FinalizeProcess() { }
  struct RuntimeInfo {
    int num_triggers_needed;
    int reference_count;
    std::mutex* mutex;
  };
  void TopDownScan(DagType& dag, const std::unordered_set<uint64_t>& start_frontier,
      const std::vector<uint64_t>& targets);
  void AppendTask(DagNode*);
  void NodeTask(DagNode*);
  void TriggerSuccessors(DagNode*);
  ThreadPool& thread_pool_;
  NodeStateMap node_states_;
  std::unordered_map<uint64_t, RuntimeInfo> rt_info_;
  int num_nodes_yet_to_finish_;
  std::mutex finish_mutex_;
  std::condition_variable finish_cond_;
};

template<typename DagType>
void DagEngine<DagType>::Process(DagType& dag, const std::vector<uint64_t>& targets) {
  PrepareProcess();
  std::unordered_set<uint64_t> start_frontier = FindStartFrontier(dag, targets);
  for(uint64_t ready_nid : node_states_.GetNodesOfState(NodeState::kReady)) {
    DagNode* ready_node = dag.GetNode(ready_nid);
    RuntimeInfo& ri = rt_info_[ready_nid];
    for(DagNode* pred : ready_node->predecessors_) {
      if(node_states_.GetState(pred->node_id()) == NodeState::kReady) {
        ++ri.num_triggers_needed;
      }
    }
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
  }
  TopDownScan(dag, start_frontier, targets);
  FinalizeProcess();
}

template<typename DagType>
void DagEngine<DagType>::OnCreateNode(DagNode* node) {
  node_states_.AddNode(node->node_id(), NodeState::kBirth);
  rt_info_[node->node_id()] = RuntimeInfo{0, 0, new std::mutex};
}

template<typename DagType>
void DagEngine<DagType>::OnDeleteNode(DagNode* node) {
  node_states_.RemoveNode(node->node_id());
}

template<typename DagType>
void DagEngine<DagType>::OnCreateEdge(DagNode* from, DagNode* to) {
  if(from->Type() == DagNode::DATA_NODE &&
      node_states_.GetState(from->node_id()) == NodeState::kCompleted) {
    typename DagType::DNode* dnode = dynamic_cast<typename DagType::DNode*>(from);
    rt_info_[from->node_id()].reference_count += 1;
  }
}

template<typename DagType>
void DagEngine<DagType>::OnIncrExternRC(typename DagType::DNode* dnode, int amount) {
  if(node_states_.GetState(dnode->node_id()) == NodeState::kCompleted) {
    RuntimeInfo& ri = rt_info_[dnode->node_id()];
    ri.reference_count += amount;
    if(ri.reference_count == 0) {
      FreeDataNodeRes(dnode);
      node_states_.ChangeState(dnode->node_id(), NodeState::kDead);
    }
  }
}

template<typename DagType>
void DagEngine<DagType>::WaitForFinish() {
  std::unique_lock<std::mutex> lck(finish_mutex_);
  while(num_nodes_yet_to_finish_ != 0)
    finish_cond_.wait(lck);
  thread_pool_.WaitForAllFinished();
}

template<typename DagType>
void DagEngine<DagType>::TopDownScan(DagType& dag, const std::unordered_set<uint64_t>& start_frontier,
    const std::vector<uint64_t>& targets) {
  num_nodes_yet_to_finish_ = node_states_.GetNodesOfState(NodeState::kReady).size();
  for(uint64_t start_nid : start_frontier) {
    AppendTask(dag.GetNode(start_nid));
  }
}

template<typename DagType>
void DagEngine<DagType>::AppendTask(DagNode* node) {
  thread_pool_.Push(std::bind(&DagEngine<DagType>::NodeTask, this, node));
}

template<typename DagType>
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

template<typename DagType>
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
  std::lock_guard<std::mutex> flck(finish_mutex_);
  if(--num_nodes_yet_to_finish_ == 0)
    finish_cond_.notify_all();
  //DLOG(INFO) << "Notify node#" << node->node_id() << " is finished";
}

template<typename DagType>
void DagEngine<DagType>::TriggerSuccessors(DagNode* node) {
  for(DagNode* succ : node->successors_) {
    RuntimeInfo& ri = rt_info_[succ->node_id()];
    std::lock_guard<std::mutex> lock(*ri.mutex);
    NodeState succ_state = node_states_.GetState(succ->node_id());
    CHECK_NE(succ_state, NodeState::kCompleted);
    CHECK_NE(succ_state, NodeState::kDead);
    if(succ_state == NodeState::kReady) {
      CHECK_GE(--ri.num_triggers_needed, 0) << "wrong #triggers for node#" << succ->node_id();
      if(ri.num_triggers_needed == 0) {
        DLOG(INFO) << "Trigger node#" << succ->node_id();
        AppendTask(succ);
      }
    }
  }
}

template<typename DagType>
void DagEngine<DagType>::GCNodes(DagType& dag) {
  std::vector<uint64_t> dead_nodes;
  for(uint64_t dead_nid : node_states_.GetNodesOfState(NodeState::kDead)) {
    dead_nodes.push_back(dead_nid);
  }
  DLOG(INFO) << dead_nodes.size() << " nodes to be GCed";
  for (uint64_t dead_nid : dead_nodes) {
    dag.DeleteNode(dead_nid);
  }
}

} // end of namespace minerva

