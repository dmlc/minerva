#include "dag_scheduler.h"
#include <vector>
#include <set>
#include <memory>
#include <glog/logging.h>
#include "system/minerva_system.h"
#include "backend/dag/dag_chunk.h"
#include "backend/dag/multi_node_lock.h"
#include "device/task.h"
#include "device/task_data.h"

using namespace std;

namespace minerva {

DagScheduler::DagScheduler(PhysicalDag* d, DeviceManager* dm) : dag_(d), dm_(dm), dispatcher_0(&DagScheduler::DispatcherRoutine, this), dispatcher_1(&DagScheduler::DispatcherRoutine, this), num_nodes_yet_to_finish_(0) {
  dm->RegisterListener(this);
}

DagScheduler::~DagScheduler() {
  WaitForAll();
  dispatcher_queue_.SignalForKill();
  dispatcher_0.join();
  dispatcher_1.join();
}

vector<BackendChunk*> DagScheduler::Create(const vector<BackendChunk*>& params,
    const std::vector<Scale>& result_sizes, shared_ptr<ComputeFn> fn) {
  auto current_device_id = MinervaSystem::Instance().current_device_id_;
  auto rst_data_nodes = Map<PhysicalDataNode*>(result_sizes, [&](const Scale& size) {
    return dag_->NewDataNode(PhysicalData(size, current_device_id, MinervaSystem::Instance().GenerateDataId()));
  });
  Iter(rst_data_nodes, [this](PhysicalDataNode* n) {
    OnCreateNode(n);
  });
  auto param_data_nodes = Map<PhysicalDataNode*>(params, [](BackendChunk* i) {
    return CHECK_NOTNULL(dynamic_cast<DagChunk*>(i))->node();
  });
  set<PhysicalDataNode*> unique_predecessors(param_data_nodes.begin(), param_data_nodes.end());
  auto ret = Map<BackendChunk*>(rst_data_nodes, [](PhysicalDataNode* n) {
    return new DagChunk(n);
  });
  {
    MultiNodeLock lock(dag_, unique_predecessors);
    auto op_node = dag_->NewOpNode(param_data_nodes, rst_data_nodes, {fn, current_device_id});
    DLOG(INFO) << "create new op node #" << op_node->node_id_ << " on device #" << current_device_id;
    OnCreateNode(op_node);
    Iter(unique_predecessors, [&](PhysicalDataNode* n) {
      OnCreateEdge(n, op_node);
    });
    Iter(rst_data_nodes, [&](PhysicalDataNode* n) {
      OnCreateEdge(op_node, n);
    });
    ProcessIfReady(op_node);
  }
  return ret;
}

void DagScheduler::Wait(BackendChunk* data) {
  unique_lock<mutex> lck(finish_mutex_);
  auto node_id = CHECK_NOTNULL(dynamic_cast<DagChunk*>(data))->node()->node_id_;
  target_ = node_id;
  while (rt_info_.GetState(node_id) != NodeState::kCompleted) {
    finish_cond_.wait(lck);
  }
  target_ = -1;
}

void DagScheduler::WaitForAll() {
  unique_lock<mutex> lck(finish_mutex_);
  /* Ensure no dumb fire of the condition variable.
   * This is really a design issue. Since we cannot keep synchronization of
   * states, we have to check for states explicitly.
   */
  CHECK_EQ(target_, -1);
  while (num_nodes_yet_to_finish_) {
    finish_cond_.wait(lck);
  }
}

shared_ptr<float> DagScheduler::GetValue(BackendChunk* chunk) {
  auto& data = CHECK_NOTNULL(dynamic_cast<DagChunk*>(chunk))->node()->data_;
  shared_ptr<float> ret(new float[data.size.Prod()], [](float* p) {
    delete[] p;
  });
  auto dev_pair = MinervaSystem::Instance().GetPtr(data.device_id, data.data_id);
  MinervaSystem::UniversalMemcpy(make_pair(Device::MemType::kCpu, ret.get()), dev_pair, data.size.Prod() * sizeof(float));
  return ret;
}

// Device listener
void DagScheduler::OnOperationComplete(Task* task) {
  dispatcher_queue_.Push({TaskType::kToComplete, task->id});
  delete task;
}

void DagScheduler::ExternRCUpdate(PhysicalDataNode* node, int delta) {
  MultiNodeLock lock(dag_, node);
  auto node_id = node->node_id_;
  auto& ri = rt_info_.At(node_id);
  node->data_.extern_rc += delta;
  switch (rt_info_.GetState(node_id)) {
    case NodeState::kCompleted: {
      // If node is in kCompleted state, that means the node has already been concretely
      // evaluated. If the node's reference count drops to zero, we could safely GC all
      // its resources.
      if (ri.reference_count == 0 && node->data_.extern_rc == 0) {
        DLOG(INFO) << "kill node #" << node->node_id_ << " during extern reference count update";
        FreeDataNodeRes(node);
        ++num_nodes_yet_to_finish_;
        dispatcher_queue_.Push({TaskType::kToDelete, node_id});
      }
      break;
    }
    case NodeState::kReady:
      break;
    default:
      LOG(FATAL) << "incorrect state for node #" << node_id;
  }
}

void DagScheduler::FreeDataNodeRes(PhysicalDataNode* node) {
  DLOG(INFO) << "free data node resource for node #" << node->node_id_ << " data #" << node->data_.data_id;
  dm_->FreeData(node->data_.data_id);
}

void DagScheduler::OnCreateNode(DagNode* node) {
  rt_info_.AddNode(node->node_id_);
}

void DagScheduler::OnDeleteNode(DagNode* node) {
  rt_info_.RemoveNode(node->node_id_);
}

void DagScheduler::OnCreateEdge(DagNode* from, DagNode* to) {
  CHECK_EQ(rt_info_.GetState(to->node_id_), NodeState::kReady) << "invalid state of node #" << to->node_id_;
  ++(rt_info_.At(from->node_id_).reference_count);
  if (rt_info_.GetState(from->node_id_) != NodeState::kCompleted) {
    ++(rt_info_.At(to->node_id_).num_triggers_needed);
  }
}

void DagScheduler::ProcessIfReady(PhysicalOpNode* target) {
  auto node_id = target->node_id_;
  CHECK_EQ(rt_info_.GetState(node_id), NodeState::kReady) << "invalid state of node #" << node_id;
  if (rt_info_.At(node_id).num_triggers_needed == 0) {
    ++num_nodes_yet_to_finish_;
    dispatcher_queue_.Push({TaskType::kToRun, node_id});
    DLOG(INFO) << "node #" << node_id << " running right after creation";
  }
}

void DagScheduler::DispatcherRoutine() {
  pair<TaskType, uint64_t> task;
  // Pop queue while not exiting
  while (!dispatcher_queue_.Pop(task)) {
    DagNode* to_delete = 0;
    {
      auto node_id = task.second;
      auto node = dag_->GetNode(node_id);
      MultiNodeLock lock(dag_, node);
      auto& ri = rt_info_.At(node_id);
      if (task.first == TaskType::kToRun && node->Type() == DagNode::NodeType::kOpNode) {  // New task to dispatch
        auto op_node = CHECK_NOTNULL(dynamic_cast<PhysicalOpNode*>(node));
        auto device_id = op_node->op_.device_id;
        Task* task = new Task();
        // The following is necessary because aggregate initialization does not use move constructors.
        Iter(op_node->inputs_, [&](PhysicalDataNode* data_node) {
          task->inputs.emplace_back(data_node->data_, data_node->node_id_);
        });
        Iter(op_node->outputs_, [&](PhysicalDataNode* data_node) {
          task->outputs.emplace_back(data_node->data_, data_node->node_id_);
        });
        task->op = op_node->op_;
        task->id = node_id;
        DLOG(INFO) << "dispatching node #" << node_id << " to device #" << device_id;
        dm_->GetDevice(device_id)->PushTask(task);
      } else if (task.first == TaskType::kToComplete ||
          (task.first == TaskType::kToRun &&
           node->Type() == DagNode::NodeType::kDataNode)) {  // Task completed
        DLOG(INFO) << "finish node #" << node_id;
        ri.state = NodeState::kCompleted;
        // Change current state and predecessors' reference counts and number of triggers
        if (node->Type() == DagNode::NodeType::kOpNode) {  // Op node
          CHECK_NE(ri.reference_count, 0) << "op node #" << node_id << " generated but not needed";
          for (auto pred : node->predecessors_) {
            auto& pred_ri = rt_info_.At(pred->node_id_);
            auto pred_node = CHECK_NOTNULL(dynamic_cast<PhysicalDataNode*>(pred));
            // Reference count decreasing to zero, not able to recover access anymore
            CHECK_EQ(pred_ri.num_triggers_needed, 0) << "#triggers incorrect for a completed data node";
            if (--pred_ri.reference_count == 0 && pred_node->data_.extern_rc == 0) {
              FreeDataNodeRes(pred_node);
              ++num_nodes_yet_to_finish_;
              dispatcher_queue_.Push({TaskType::kToDelete, pred_node->node_id_});
            }
          }
        } else {  // Data node
          auto data_node = CHECK_NOTNULL(dynamic_cast<PhysicalDataNode*>(node));
          // Data node generated but not needed
          if (ri.reference_count == 0 && data_node->data_.extern_rc == 0) {
            FreeDataNodeRes(data_node);
            ++num_nodes_yet_to_finish_;
            dispatcher_queue_.Push({TaskType::kToDelete, data_node->node_id_});
          }
          CHECK_EQ(node->predecessors_.size(), 1) << "data node should have no more than one predecessor";
          auto pred_node = *node->predecessors_.begin();
          auto& pred_ri = rt_info_.At(pred_node->node_id_);
          CHECK_EQ(pred_ri.num_triggers_needed, 0) << "#triggers incorrect for a completed op node";
          if (--pred_ri.reference_count == 0) {
            ++num_nodes_yet_to_finish_;
            dispatcher_queue_.Push({TaskType::kToDelete, pred_node->node_id_});
          }
        }
        // Trigger successors
        for (auto succ : node->successors_) {
          auto& ri = rt_info_.At(succ->node_id_);
          --ri.num_triggers_needed;
          if (ri.state == NodeState::kReady && ri.num_triggers_needed == 0) {
            DLOG(INFO) << "trigger node #" << succ->node_id_;
            ++num_nodes_yet_to_finish_;
            dispatcher_queue_.Push({TaskType::kToRun, succ->node_id_});
          }
        }
        DecrNumNodesYetToFinish(node_id);
      } else if (task.first == TaskType::kToDelete) {
        CHECK_EQ(ri.state, NodeState::kCompleted);
        DLOG(INFO) << "dispatcher ready to delete node #" << node_id;
        OnDeleteNode(node);
        to_delete = dag_->RemoveNodeFromDag(node_id);
        DecrNumNodesYetToFinish(node_id);
      } else {
        LOG(FATAL) << "illegal task state";
      }
    }
    delete to_delete;
  }
}

void DagScheduler::DecrNumNodesYetToFinish(uint64_t node_id) {
  --num_nodes_yet_to_finish_;
  {
    unique_lock<mutex> lck(finish_mutex_);
    if (num_nodes_yet_to_finish_ == 0 || node_id == target_) {
      finish_cond_.notify_all();
    }
  }
}

}  // namespace minerva

