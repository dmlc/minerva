#include "procedures/dag_scheduler.h"
#include <queue>
#include <list>
#include <glog/logging.h>

using namespace std;

namespace minerva {

DagScheduler::DagScheduler(PhysicalDag* d) : dispatcher_(&DagScheduler::DispatcherRoutine, this), num_nodes_yet_to_finish_(0) {
  dag_ = d;
}

void DagScheduler::WaitForFinish() {
  unique_lock<mutex> lck(finish_mutex_);
  while (num_nodes_yet_to_finish_) {
    finish_cond_.wait(lck);
  }
}

void DagScheduler::GCNodes() {
  auto dead_set = rt_info_.dead_nodes();
  DLOG(INFO) << dead_set.size() << " nodes to be GCed";
  for (auto id :dead_set) {
    dag_->DeleteNode(id);
  }
}

void DagScheduler::OnIncrExternRC(PhysicalDataNode* node, int amount) {
  switch (rt_info_.GetState(node->node_id())) {
    case NodeState::kReady:
    case NodeState::kRunning:
    case NodeState::kCompleted: {
      auto& ri = rt_info_.At(node->node_id());
      ri.reference_count.fetch_add(amount);
      if (ri.reference_count == 0) {
        FreeDataNodeRes(node);
        ri.state = NodeState::kDead;
        rt_info_.KillNode(node->node_id());
      }
      break;
    }
    default:
      break;
  }
}

void DagScheduler::OnCreateNode(DagNode* node) {
  rt_info_.AddNode(node->node_id());
}

void DagScheduler::OnDeleteNode(DagNode* node) {
  rt_info_.RemoveNode(node->node_id());
}

void DagScheduler::OnCreateEdge(DagNode* from, DagNode*) {
  if (from->Type() == DagNode::NodeType::kDataNode) {
    switch (rt_info_.GetState(from->node_id())) {
      case NodeState::kReady:
      case NodeState::kRunning:
      case NodeState::kCompleted:
        ++(rt_info_.At(from->node_id()).reference_count);
        break;
      default:
        break;
    }
  }
}

void DagScheduler::Process(const vector<uint64_t>& targets) {
  // Nodes in `queue` should all have state `kBirth`
  queue<uint64_t> queue;
  for (auto id : targets) {
    // `targets` should consist of only data nodes
    CHECK_EQ(dag_->GetNode(id)->Type(), DagNode::NodeType::kDataNode);
    switch (rt_info_.GetState(id)) {
      case NodeState::kBirth:
        queue.push(id);
        break;
      case NodeState::kDead:
        CHECK(false) << "invalid node state of id " << id;
        break;
      default:
        break;
    }
  }
  while (!queue.empty()) {
    auto node_id = queue.front();
    auto node = dag_->GetNode(node_id);
    auto& ri = rt_info_.At(node_id);
    queue.pop();
    list<lock_guard<mutex>> lcks;
    if (node->Type() == DagNode::NodeType::kOpNode) {
      for (auto pred : node->predecessors_) {
        // Lock preceding data nodes to prevent premature trigger
        lcks.emplace_back(rt_info_.At(pred->node_id()).m);
      }
    }
    ri.state = NodeState::kReady;
    for (auto pred : node->predecessors_) {
      switch (rt_info_.GetState(pred->node_id())) {
        case NodeState::kBirth:
          queue.push(pred->node_id());
        case NodeState::kReady:
        case NodeState::kRunning:
          // Set triggers count
          ++ri.num_triggers_needed;
          break;
        case NodeState::kCompleted:
          break;
        default:
          CHECK(false) << "invalid node state of id " << pred->node_id() << " on dependency path";
      }
    }
    if (node->Type() == DagNode::NodeType::kOpNode) {
      for (auto succ : node->successors_) {
        if (CalcTotalReferenceCount(dynamic_cast<PhysicalDataNode*>(succ)) == 0) {
          rt_info_.At(succ->node_id()).state = NodeState::kDead;
          rt_info_.KillNode(succ->node_id());
        } else {
          rt_info_.At(succ->node_id()).state = NodeState::kReady;
        }
      }
      ri.reference_count = -1;
    } else {
      ri.reference_count = CalcTotalReferenceCount(dynamic_cast<PhysicalDataNode*>(node));
    }
    if (ri.num_triggers_needed == 0) {
      DLOG(INFO) << "starting node id " << node_id;
      ri.state = NodeState::kRunning;
      ++num_nodes_yet_to_finish_;
      dispatcher_queue_.Push(node_id);
    }
  }
}

void DagScheduler::OnOperationComplete(uint64_t id) {
  auto node = dag_->GetNode(id);
  auto& ri = rt_info_.At(id);
  lock_guard<mutex> lck(ri.m);
  // Change current state and predecessors' reference counts
  if (node->Type() == DagNode::NodeType::kOpNode) {
    for (auto pred : node->predecessors_) {
      auto& pred_ri = rt_info_.At(pred->node_id());
      // Reference count decreasing to zero, not able to recover access anymore
      if (--pred_ri.reference_count == 0) {
        FreeDataNodeRes(dynamic_cast<PhysicalDataNode*>(pred));
        // No locks needed, since `reference_count` cannot be decreased to 0 multiple times
        pred_ri.state = NodeState::kDead;
        rt_info_.KillNode(pred->node_id());
      }
    }
    ri.state = NodeState::kDead;
    rt_info_.KillNode(id);
  } else {
    ri.state = NodeState::kCompleted;
  }
  // Trigger successors
  for (auto succ : node->successors_) {
    auto& ri = rt_info_.At(succ->node_id());
    if (ri.state == NodeState::kReady) {
      if (--ri.num_triggers_needed == 0) {
        DLOG(INFO) << "trigger node id " << succ->node_id();
        ri.state = NodeState::kRunning;
        ++num_nodes_yet_to_finish_;
        dispatcher_queue_.Push(succ->node_id());
      }
    }
  }
  {
    lock_guard<mutex> lck(finish_mutex_);
    if (--num_nodes_yet_to_finish_ == 0) {
      finish_cond_.notify_all();
    }
  }
}

int DagScheduler::CalcTotalReferenceCount(PhysicalDataNode* node) {
  int count = node->data_.extern_rc;
  for (auto succ : node->successors_) {
    switch (rt_info_.GetState(succ->node_id())) {
      case NodeState::kBirth:
      case NodeState::kReady:
        ++count;
        break;
      default:
        CHECK(false) << "invalid node state id " << succ->node_id();
        break;
    }
  }
  return count;
}

void FreeDataNodeRes(PhysicalDataNode* node) {
  // TODO Notify device to free data storage
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

