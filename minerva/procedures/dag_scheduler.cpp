#include "procedures/dag_scheduler.h"
#include <queue>
#include <glog/logging.h>

using namespace std;

namespace minerva {

DagScheduler::DagScheduler(PhysicalDag* d) : dispatcher_(&DagScheduler::DispatcherRoutine, this) {
  dag_ = d;
}

void DagScheduler::WaitForFinish() {
  unique_lock<mutex> lck(finish_mutex_);
  while (num_nodes_yet_to_finish_) {
    finish_cond_.wait(lck);
  }
}

void DagScheduler::GCNodes() {
  lock_guard<mutex> lck(scheduler_busy_);
  auto& dead_set = node_states_.GetNodesOfState(NodeState::kDead);
  vector<uint64_t> dead_nodes(dead_set.begin(), dead_set.end());
  DLOG(INFO) << dead_nodes.size() << " nodes to be GCed";
  for (auto id : dead_nodes) {
    dag_->DeleteNode(id);
  }
}

void DagScheduler::OnIncrExternRC(PhysicalDataNode* node, int amount) {
  if (node_states_.GetState(node->node_id()) == NodeState::kCompleted) {
    auto& ri = rt_info_[node->node_id()];
    ri.reference_count += amount;
    if (ri.reference_count == 0) {
      FreeDataNodeRes(node);
      node_states_.ChangeState(node->node_id(), NodeState::kDead);
    }
  }
}

void DagScheduler::OnCreateNode(DagNode* node) {
  node_states_.AddNode(node->node_id(), NodeState::kBirth);
}

void DagScheduler::OnDeleteNode(DagNode* node) {
  node_states_.RemoveNode(node->node_id());
}

void DagScheduler::OnCreateEdge(DagNode* from, DagNode* to) {
  if (from->Type() == DagNode::NodeType::kDataNode && node_states_.GetState(from->node_id()) == NodeState::kCompleted) {
    rt_info_[from->node_id()].reference_count += 1;
  }
}

void DagScheduler::Process(const vector<uint64_t>& targets) {
  lock_guard<mutex> lck(scheduler_busy_);
  auto start_frontier = FindStartFrontier(targets);
  for (auto ready_id : node_states_.GetNodesOfState(NodeState::kReady)) {
    auto ready_node = dag_->GetNode(ready_id);
    RuntimeInfo& ri = rt_info_[ready_id];
    for (auto pred : ready_node->predecessors_) {
      if (node_states_.GetState(pred->node_id()) == NodeState::kReady) {
        ++ri.num_triggers_needed;
      }
    }
    if (ready_node->Type() == DagNode::NodeType::kOpNode) {
      for (auto ready_op_succ : ready_node->successors_) {
        auto succ_dnode = dynamic_cast<PhysicalDataNode*>(ready_op_succ);
        if (CalcTotalReferenceCount(succ_dnode) == 0) {
          node_states_.ChangeState(ready_op_succ->node_id(), NodeState::kDead);
        } else {
          node_states_.ChangeState(ready_op_succ->node_id(), NodeState::kReady);
        }
      }
      ri.reference_count = -1;
    } else {
      auto ready_dnode = dynamic_cast<PhysicalDataNode*>(ready_node);
      ri.reference_count = CalcTotalReferenceCount(ready_dnode);
    }
  }
  num_nodes_yet_to_finish_ = node_states_.GetNodesOfState(NodeState::kReady).size();
  for (auto id : start_frontier) {
    dispatcher_queue_.Push(id);
  }
}

void DagScheduler::OnOperationComplete(uint64_t id) {
  lock_guard<mutex> lck(scheduler_busy_);
  auto node = dag_->GetNode(id);
  if (node->Type() == DagNode::NodeType::kOpNode) {
    for (auto pred : node->predecessors_) {
      auto& pred_ri = rt_info_[pred->node_id()];
      if (--pred_ri.reference_count == 0) {
        FreeDataNodeRes(dynamic_cast<PhysicalDataNode*>(pred));
        node_states_.ChangeState(pred->node_id(), NodeState::kDead);
      }
    }
  } else {
    node_states_.ChangeState(id, NodeState::kCompleted);
  }
  for (auto succ : node->successors_) {
    auto& ri = rt_info_[succ->node_id()];
    auto succ_state = node_states_.GetState(succ->node_id());
    if (succ_state == NodeState::kReady) {
      if (--ri.num_triggers_needed == 0) {
        DLOG(INFO) << "trigger node id " << succ->node_id();
        dispatcher_queue_.Push(succ->node_id());
      }
    }
  }
  {
    lock_guard<mutex> lck2(finish_mutex_);
    if (--num_nodes_yet_to_finish_ == 0) {
      finish_cond_.notify_all();
    }
  }
}

int DagScheduler::CalcTotalReferenceCount(PhysicalDataNode* node) const {
  int count = node->data_.extern_rc;
  for (auto succ : node->successors_) {
    auto succ_state = node_states_.GetState(succ->node_id());
    if (succ_state == NodeState::kBirth || succ_state == NodeState::kReady) {
     ++count;
    }
  }
  return count;
}

void FreeDataNodeRes(PhysicalDataNode* node) {
  // TODO Notify device to free data storage
}

unordered_set<uint64_t> DagScheduler::FindStartFrontier(const std::vector<uint64_t>& targets) {
  // Assert: `targets` all be living data nodes
  unordered_set<uint64_t> start_frontier;
  queue<uint64_t> queue;
  for (auto id : targets) {
    if (node_states_.GetState(id) != NodeState::kCompleted) {
      queue.push(id);
    }
  }
  while (!queue.empty()) {
    auto node_id = queue.front();
    auto node = dag_->GetNode(node_id);
    queue.pop();
    node_states_.ChangeState(node_id, NodeState::kReady);
    int pred_count = 0;
    for (auto pred : node->predecessors_) {
      auto pred_state = node_states_.GetState(pred->node_id());
      switch (pred_state) {
        case NodeState::kBirth:
          queue.push(pred->node_id());
        case NodeState::kReady:
          ++pred_count;
          break;
        case NodeState::kCompleted:
          break;
        default:
          CHECK(false) << "invalid node state of id " << pred->node_id() << " on dependency path";
      }
    }
    if (pred_count == 0) {
      start_frontier.insert(node_id);
    }
  }
  return start_frontier;
}

void DagScheduler::DispatcherRoutine() {
  uint64_t node_id;
  // Pop queue while not exiting
  while (!dispatcher_queue_.Pop(node_id)) {
    DLOG(INFO) << "dispatching node id " << node_id;
    // TODO dispatch to some device
  }
}

}  // namespace minerva

