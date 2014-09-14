#include "procedures/dag_scheduler.h"
#include <queue>
#include <glog/logging.h>

using namespace std;

namespace minerva {

DagScheduler::DagScheduler(PhysicalDag* d) {
  dag_ = d;
}

void DagScheduler::WaitForFinish() {
  unique_lock<mutex> lck(finish_mutex_);
  while (num_nodes_yet_to_finish_) {
    finish_cond_.wait(lck);
  }
}

int DagScheduler::CalcTotalReferenceCount(PhysicalDataNode* node) {
  int count = node->data_.extern_rc;
  for (auto succ : node->successors_) {
    auto succ_state = node_states_.GetState(succ->node_id());
    if (succ_state == NodeState::kBirth || succ_state == NodeState::kReady) {
     ++count;
    }
  }
  return count;
}

void DagScheduler::Process(const vector<uint64_t>& targets) {
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
  TopDownScan(start_frontier, targets);
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

}  // namespace minerva
