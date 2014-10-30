#include "procedures/dag_scheduler.h"
#include <queue>
#include <list>
#include <glog/logging.h>
#include "system/minerva_system.h"

using namespace std;

namespace minerva {

DagScheduler::DagScheduler(PhysicalDag* d) : dispatcher_(&DagScheduler::DispatcherRoutine, this), num_nodes_yet_to_finish_(0) {
  dag_ = d;
}

DagScheduler::~DagScheduler() {
  dispatcher_queue_.SignalForKill();
  dispatcher_.join();
}

void DagScheduler::WaitForFinish() {
  unique_lock<mutex> lck(finish_mutex_);
  while (num_nodes_yet_to_finish_) {
    finish_cond_.wait(lck);
  }
}

void DagScheduler::GCNodes() {
  lock_guard<recursive_mutex> lck(m_);
  auto dead_set = rt_info_.dead_nodes();
  DLOG(INFO) << dead_set.size() << " nodes to be GCed";
  for (auto id :dead_set) {
    dag_->DeleteNode(id);
  }
}

void DagScheduler::OnExternRCUpdate(PhysicalDataNode* node) {
  lock_guard<recursive_mutex> lck(m_);
  switch (rt_info_.GetState(node->node_id())) {
    case NodeState::kCompleted: {
      auto& ri = rt_info_.At(node->node_id());
      if (ri.reference_count == 0 && node->data_.extern_rc == 0) {
        FreeDataNodeRes(node);
        ri.state = NodeState::kDead;
        rt_info_.KillNode(node->node_id());
        DLOG(INFO) << "GC node #" << node->node_id() << " during extern reference count update";
      }
      break;
    }
    case NodeState::kBirth: {
      auto& ri = rt_info_.At(node->node_id());
      if (ri.reference_count == 0 && node->data_.extern_rc == 0) {
        queue<PhysicalOpNode*> probably_unused_nodes;
        CHECK_EQ(node->predecessors_.size(), 1) << "data node have more than one predecessors";
        probably_unused_nodes.push(CHECK_NOTNULL(dynamic_cast<PhysicalOpNode*>(*node->predecessors_.begin())));
        while (probably_unused_nodes.size()) {
          auto node = probably_unused_nodes.front();
          auto& node_ri = rt_info_.At(node->node_id());
          probably_unused_nodes.pop();
          if (node_ri.state == NodeState::kDead) {
            // Already GCed through some other path
            continue;
          }
          CHECK_EQ(node_ri.state, NodeState::kBirth);
          bool not_used = true;
          // Op node is not used if all successors are not used
          for (auto i : node->successors_) {
            auto dnode = CHECK_NOTNULL(dynamic_cast<PhysicalDataNode*>(i));
            auto& dnode_ri = rt_info_.At(dnode->node_id());
            if (dnode_ri.reference_count != 0 || dnode->data_.extern_rc != 0) {
              not_used = false;
              break;
            }
          }
          if (not_used) {
            for (auto i : node->successors_) {
              auto dnode = CHECK_NOTNULL(dynamic_cast<PhysicalDataNode*>(i));
              auto& dnode_ri = rt_info_.At(dnode->node_id());
              CHECK_EQ(dnode_ri.state, NodeState::kBirth);
              dnode_ri.state = NodeState::kDead;
              dnode_ri.num_triggers_needed = 0;
              rt_info_.KillNode(dnode->node_id());
              DLOG(INFO) << "GC node #" << dnode->node_id() << " during extern reference count update";
            }
            node_ri.state = NodeState::kDead;
            node_ri.num_triggers_needed = 0;
            node_ri.reference_count = 0;
            rt_info_.KillNode(node->node_id());
            DLOG(INFO) << "GC node #" << node->node_id() << " during extern reference count update";
            for (auto i : node->predecessors_) {
              auto dnode = CHECK_NOTNULL(dynamic_cast<PhysicalDataNode*>(i));
              auto& dnode_ri = rt_info_.At(dnode->node_id());
              CHECK_NE(dnode_ri.state, NodeState::kDead);
              --dnode_ri.reference_count;
              if (dnode_ri.state == NodeState::kBirth && dnode_ri.reference_count == 0 && dnode->data_.extern_rc == 0) {
                CHECK_EQ(dnode->predecessors_.size(), 1) << "data node have more than one predecessors";
                probably_unused_nodes.push(CHECK_NOTNULL(dynamic_cast<PhysicalOpNode*>(*dnode->predecessors_.begin())));
              } else if (dnode_ri.state == NodeState::kCompleted && dnode_ri.reference_count == 0 && dnode->data_.extern_rc == 0) {
                FreeDataNodeRes(dnode);
                dnode_ri.state = NodeState::kDead;
                rt_info_.KillNode(dnode->node_id());
                DLOG(INFO) << "GC node #" << dnode->node_id() << " during extern reference count update";
              }
            }
          }
        }
      }
    }
    case NodeState::kReady:
      break;
    default:
      CHECK(false) << "incorrect state for node #" << node->node_id();
  }
}

void DagScheduler::OnCreateNode(DagNode* node) {
  lock_guard<recursive_mutex> lck(m_);
  rt_info_.AddNode(node->node_id());
}

void DagScheduler::OnDeleteNode(DagNode* node) {
  // Forced deletion of nodes will not handle runtime information
  lock_guard<recursive_mutex> lck(m_);
  if (node->Type() == DagNode::NodeType::kDataNode) {
    FreeDataNodeRes(CHECK_NOTNULL(dynamic_cast<PhysicalDataNode*>(node)));
  }
  rt_info_.RemoveNode(node->node_id());
}

void DagScheduler::OnCreateEdge(DagNode* from, DagNode* to) {
  lock_guard<recursive_mutex> lck(m_);
  CHECK_NE(rt_info_.GetState(from->node_id()), NodeState::kDead) << "invalid state of node #" << from->node_id();
  CHECK_EQ(rt_info_.GetState(to->node_id()), NodeState::kBirth) << "invalid state of node #" << to->node_id();
  ++(rt_info_.At(from->node_id()).reference_count);
  if (rt_info_.GetState(from->node_id()) != NodeState::kCompleted) {
    ++(rt_info_.At(to->node_id()).num_triggers_needed);
  }
}

void DagScheduler::OnBeginModify() {
  m_.lock();
}

void DagScheduler::OnFinishModify() {
  m_.unlock();
}

// Device listener
void DagScheduler::OnOperationComplete(uint64_t id) {
  dispatcher_queue_.Push({TaskType::kToComplete, id});
}

void DagScheduler::Process(const vector<uint64_t>& targets) {
  // Nodes in `queue` should all have state `kBirth`
  lock_guard<recursive_mutex> lck(m_);
  queue<uint64_t> queue;
  for (auto id : targets) {
    // `targets` should consist of only data nodes
    CHECK_EQ(static_cast<int>(dag_->GetNode(id)->Type()), static_cast<int>(DagNode::NodeType::kDataNode));
    switch (rt_info_.GetState(id)) {
      case NodeState::kBirth:
        queue.push(id);
        rt_info_.At(id).state = NodeState::kReady;
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
    for (auto pred : node->predecessors_) {
      switch (rt_info_.GetState(pred->node_id())) {
        case NodeState::kBirth:
          queue.push(pred->node_id());
          rt_info_.At(pred->node_id()).state = NodeState::kReady;
          break;
        case NodeState::kReady:
          break;
        case NodeState::kCompleted:
          break;
        default:
          CHECK(false) << "invalid state of node #" << pred->node_id() << " on dependency path";
      }
    }
    if (node->Type() == DagNode::NodeType::kOpNode) {
      for (auto succ : node->successors_) {
        rt_info_.At(succ->node_id()).state = NodeState::kReady;
      }
    }
    if (ri.num_triggers_needed == 0) {
      DLOG(INFO) << "starting from node #" << node_id;
      ++num_nodes_yet_to_finish_;
      dispatcher_queue_.Push({TaskType::kToRun, node_id});
    }
  }
}

void DagScheduler::FreeDataNodeRes(PhysicalDataNode* node) {
  DLOG(INFO) << "free data node resource for node #" << node->node_id() << " data #" << node->data_.data_id;
  MinervaSystem::Instance().device_manager().FreeData(node->data_.data_id);
}

void DagScheduler::DispatcherRoutine() {
  pair<TaskType, uint64_t> task;
  // Pop queue while not exiting
  while (!dispatcher_queue_.Pop(task)) {
    lock_guard<recursive_mutex> lck(m_);
    auto node_id = task.second;
    auto node = dag_->GetNode(node_id);
    auto& ri = rt_info_.At(node_id);
    bool finish_directly = false;
    if (task.first == TaskType::kToRun) {  // New task to dispatch
      uint64_t device_id;
      if (node->Type() == DagNode::NodeType::kOpNode) {
        device_id = CHECK_NOTNULL(dynamic_cast<PhysicalOpNode*>(node))->op_.compute_fn->device_id;
        DLOG(INFO) << "dispatching node #" << node_id << " to device " << device_id;
        MinervaSystem::Instance().device_manager().GetDevice(device_id)->PushTask(node_id);
      } else {
        finish_directly = true;
      }
    }
    if (task.first == TaskType::kToComplete || finish_directly) {  // Task completed
      DLOG(INFO) << "finishing node #" << node_id;
      ri.state = NodeState::kCompleted;
      // Change current state and predecessors' reference counts and #triggers
      if (node->Type() == DagNode::NodeType::kOpNode) {  // Op node
        CHECK_NE(ri.reference_count, 0) << "op node generated but not needed";
        for (auto pred : node->predecessors_) {
          auto& pred_ri = rt_info_.At(pred->node_id());
          auto pred_node = CHECK_NOTNULL(dynamic_cast<PhysicalDataNode*>(pred));
          // Reference count decreasing to zero, not able to recover access anymore
          CHECK_EQ(pred_ri.num_triggers_needed, 0) << "#triggers incorrect for a completed data node";
          if (--pred_ri.reference_count == 0 && pred_node->data_.extern_rc == 0) {
            FreeDataNodeRes(pred_node);
            // No locks needed, since `reference_count` cannot be decreased to 0 multiple times
            pred_ri.state = NodeState::kDead;
            rt_info_.KillNode(pred->node_id());
          }
        }
      } else {  // Data node
        auto data_node = CHECK_NOTNULL(dynamic_cast<PhysicalDataNode*>(node));
        // Data node generated but not needed
        if (ri.reference_count == 0 && data_node->data_.extern_rc == 0) {
          FreeDataNodeRes(data_node);
          ri.state = NodeState::kDead;
          rt_info_.KillNode(node_id);
        }
        CHECK_EQ(node->predecessors_.size(), 1) << "data node have more than one predecessors";
        auto pred_node = *node->predecessors_.begin();
        auto& pred_ri = rt_info_.At(pred_node->node_id());
        CHECK_EQ(pred_ri.num_triggers_needed, 0) << "#triggers incorrect for a completed op node";
        if (--pred_ri.reference_count == 0) {
          pred_ri.state = NodeState::kDead;
          rt_info_.KillNode(pred_node->node_id());
        }
      }
      // Trigger successors
      {
        for (auto succ : node->successors_) {
          auto& ri = rt_info_.At(succ->node_id());
          --ri.num_triggers_needed;
          if (ri.state == NodeState::kReady) {
            if (ri.num_triggers_needed == 0) {
              DLOG(INFO) << "trigger node #" << succ->node_id();
              ++num_nodes_yet_to_finish_;
              dispatcher_queue_.Push({TaskType::kToRun, succ->node_id()});
            }
          }
        }
      }
      --num_nodes_yet_to_finish_;
    }
    {
      lock_guard<mutex> lck(finish_mutex_);
      if (num_nodes_yet_to_finish_ == 0) {
        finish_cond_.notify_all();
      }
    }
  }
}

}  // namespace minerva

