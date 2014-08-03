#include <random>
#include <queue>
#include <glog/logging.h>

#include "procedures/physical_engine.h"
#include "op/physical.h"
#include "system/minerva_system.h"

#define THREAD_NUM 4

using namespace std;

namespace minerva {

static MinervaSystem& ms = MinervaSystem::Instance();

PhysicalEngine::PhysicalEngine(): thread_pool_(THREAD_NUM, this) {
  Init();
}

PhysicalEngine::~PhysicalEngine() {
  task_queue_.SignalForKill();
}

void PhysicalEngine::Process(PhysicalDag&, const std::vector<uint64_t>& targets) {
  // TODO Ignoring PhysicalDag, use MinervaSystem instead
  auto ready_to_execute = FindRootNodes(targets);
  for (auto i: ready_to_execute) {
    AppendTask(i, bind(&PhysicalEngine::NodeRunner, this, placeholders::_1));
  }
  {
    // Waiting execution to complete
    for (auto i: targets) {
      LOG(INFO) << "Wait for node (id=" << i << ") finish.";
      unique_lock<mutex> lock(node_states_mutex_);
      if (node_states_[i].state != NodeState::kComplete) {
        node_states_[i].on_complete->wait(lock);
      }
      delete node_states_[i].on_complete;
      node_states_[i].on_complete = 0;
      LOG(INFO) << "Node (id=" << i << ") complete.";
    }
  }
}

void PhysicalEngine::Init() {
  // Then we can load user defined runners
}

void PhysicalEngine::OnCreateNode(DagNode* node) {
  lock_guard<mutex> lock(node_states_mutex_);
  NodeState ns{NodeState::kNoNeed, 0, NULL};
  node_states_.insert(make_pair(node->node_id(), ns));
}

void PhysicalEngine::OnDeleteNode(DagNode* node) {
  lock_guard<mutex> lock(node_states_mutex_);
  node_states_.erase(node->node_id());
}

unordered_set<DagNode*> PhysicalEngine::FindRootNodes(const vector<uint64_t>& targets) {
  lock_guard<mutex> lock(node_states_mutex_);
  auto& dag = ms.physical_dag();
  queue<uint64_t> ready_node_queue;
  unordered_set<DagNode*> ready_to_execute;
  for (auto i: targets) {
    // Don't push complete nodes
    if (node_states_[i].state != NodeState::kComplete) {
      ready_node_queue.push(i);
    }
  }
  while (!ready_node_queue.empty()) {
    uint64_t curid = ready_node_queue.front();
    ready_node_queue.pop();
    auto& it = node_states_[curid];
    auto node = dag.GetNode(curid);;
    it.state = NodeState::kReady;
    it.dependency_counter = 0;
    for (auto i: node->predecessors_) {
      switch (node_states_[i->node_id()].state) {
        // Count dependency and recursively search predecessors
        case NodeState::kNoNeed:
          ready_node_queue.push(i->node_id());
        case NodeState::kReady:
          ++it.dependency_counter;
          break;
        default:
          break;
      }
    }
    // All successors of OpNode will be set ready. Successor of an incomplete OpNode could not be complete.
    if (node->Type() == DagNode::OP_NODE) {
      for (auto i: node->successors_) {
        node_states_[i->node_id()].state = NodeState::kReady;
      }
    }
    // All predecessors are complete, or there are no predecessors at all
    if (!it.dependency_counter) {
      ready_to_execute.insert(node);
    }
  }
  for (auto i: targets) {
    if (node_states_[i].state != NodeState::kComplete) {
      node_states_[i].on_complete = new condition_variable;
    }
  }
  return ready_to_execute;
}

void PhysicalEngine::NodeRunner(DagNode* node) {
  if (node->Type() == DagNode::OP_NODE) { // OpNode
    vector<DataShard> input;
    vector<DataShard> output;
    PhysicalOpNode* phy_op_node = dynamic_cast<PhysicalOpNode*>(node);
    for (auto n: phy_op_node->inputs_) {
      PhysicalData& in_data = dynamic_cast<PhysicalDataNode*>(n)->data_;
      input.push_back(DataShard(in_data));
    }
    for (auto n: phy_op_node->outputs_) { // Allocate storage for all outputs
      PhysicalData& out_data = dynamic_cast<PhysicalDataNode*>(n)->data_;
      int rc = out_data.extern_rc + n->successors_.size();
      ms.data_store().CreateData(out_data.data_id, DataStore::CPU, out_data.size.Prod(), rc);
      output.push_back(DataShard(out_data));
    }
    // call compute function
    PhysicalOp& op = phy_op_node->op_;
    LOG(INFO) << "Execute node#" << node->node_id() << " compute fn: " << op.compute_fn->Name();
    op.compute_fn->Execute(input, output, BASIC); // TODO decide impl_type
    for (auto n: phy_op_node->predecessors_) {// de-refer predecessor's data
      PhysicalData& in_data = dynamic_cast<PhysicalDataNode*>(n)->data_;
      ms.data_store().DecrReferenceCount(in_data.data_id);
    }
  } 
  // trigger successors
  {
    lock_guard<mutex> lock(node_states_mutex_);
    auto succ = node->successors_;
    for (auto i: succ) {
      auto state = node_states_.find(i->node_id());
      if (state == node_states_.end()) { // New nodes, not committed yet
        continue;
      }
      // Append node if all predecessors are finished
      if (state->second.state == NodeState::kReady && !(--state->second.dependency_counter)) {
        AppendTask(i, bind(&PhysicalEngine::NodeRunner, this, placeholders::_1));
      }
    }
    node_states_[node->node_id()].state = NodeState::kComplete;
    if (node_states_[node->node_id()].on_complete) {
      //printf("Target complete %u\n", (unsigned int) dynamic_cast<PhysicalDataNode*>(node)->data_.data_id);
      node_states_[node->node_id()].on_complete->notify_all();
    }
  }
}

void PhysicalEngine::AppendTask(Task node, Callback callback) {
  task_queue_.Push(make_pair(node, callback));
}

bool PhysicalEngine::GetNewTask(thread::id id, TaskPair& task) {
  return task_queue_.Pop(task);
}

} // end of namespace minerva
