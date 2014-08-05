#include <random>
#include <queue>
#include <glog/logging.h>

#include "procedures/physical_engine.h"
#include "op/physical.h"
#include "system/minerva_system.h"

#define THREAD_NUM 4

using namespace std;

namespace minerva {

PhysicalEngine::PhysicalEngine(NodeStateMap<PhysicalDag>& ns):
  node_states_(ns), thread_pool_(THREAD_NUM, this) {
  Init();
}

PhysicalEngine::~PhysicalEngine() {
  task_queue_.SignalForKill();
}

void PhysicalEngine::Process(PhysicalDag& dag, const std::vector<uint64_t>& targets) {
  cout << dag.PrintDag() << endl;
  GCNodes(dag);
  auto ready_to_execute = FindRootNodes(dag, targets);
  for (auto i: ready_to_execute) {
    AppendTask(i, bind(&PhysicalEngine::NodeRunner, this, placeholders::_1));
  }
  {
    // Waiting execution to complete
    for (auto tgtid: targets) {
      LOG(INFO) << "Wait for node (id=" << tgtid << ") finish.";
      unique_lock<mutex> lock(node_states_mutex_);
      if (node_states_.GetState(tgtid) != NodeState::kCompleted) {
        rt_states_[tgtid].on_complete->wait(lock);
      }
      delete rt_states_[tgtid].on_complete;
      rt_states_[tgtid].on_complete = nullptr;
      LOG(INFO) << "Node (id=" << tgtid << ") complete.";
    }
  }
  GCNodes(dag);
}

void PhysicalEngine::Init() {
  // Then we can load user defined runners
}

void PhysicalEngine::OnCreateNode(DagNode* node) {
  lock_guard<mutex> lock(node_states_mutex_);
  RuntimeState ns{0, NULL};
  rt_states_.insert(make_pair(node->node_id(), ns));
}

void PhysicalEngine::OnDeleteNode(DagNode* node) {
  lock_guard<mutex> lock(node_states_mutex_);
  rt_states_.erase(node->node_id());
}

unordered_set<DagNode*> PhysicalEngine::FindRootNodes(PhysicalDag& dag, const vector<uint64_t>& targets) {
  lock_guard<mutex> lock(node_states_mutex_);
  queue<uint64_t> ready_node_queue;
  unordered_set<DagNode*> ready_to_execute;
  for (uint64_t tgtid: targets) {
    // Don't push complete nodes
    if (node_states_.GetState(tgtid) != NodeState::kCompleted) {
      ready_node_queue.push(tgtid);
    }
  }
  while (!ready_node_queue.empty()) {
    uint64_t curid = ready_node_queue.front();
    ready_node_queue.pop();
    node_states_.ChangeState(curid, NodeState::kReady);
    RuntimeState& rts = rt_states_[curid];
    auto node = dag.GetNode(curid);;
    rts.dependency_counter = 0;
    for (auto i: node->predecessors_) {
      switch (node_states_.GetState(curid)) {
        // Count dependency and recursively search predecessors
        case NodeState::kBirth:
          ready_node_queue.push(i->node_id());
        case NodeState::kReady:
          ++rts.dependency_counter;
          break;
        default:
          break;
      }
    }
    // All successors of OpNode will be set ready. Successor of an incomplete OpNode could not be complete.
    if (node->Type() == DagNode::OP_NODE) {
      for (auto i: node->successors_) {
        node_states_.ChangeState(i->node_id(), NodeState::kReady);
      }
    }
    // All predecessors are complete, or there are no predecessors at all
    if (rts.dependency_counter == 0) {
      ready_to_execute.insert(node);
    }
  }
  for (uint64_t tgtid: targets) {
    if (node_states_.GetState(tgtid) != NodeState::kCompleted) {
      rt_states_[tgtid].on_complete = new condition_variable;
    }
  }
  return ready_to_execute;
}

void PhysicalEngine::GCNodes(PhysicalDag& dag) {
  for(uint64_t nid : node_states_.GetNodesOfState(NodeState::kCompleted)) {
    DagNode* node = dag.GetNode(nid);
    switch(node->Type()) {
    case DagNode::OP_NODE:
      node_states_.ChangeState(nid, NodeState::kDead);// op nodes are just GCed
      break;
    case DagNode::DATA_NODE:
      PhysicalDataNode* dnode = dynamic_cast<PhysicalDataNode*>(node);
      int dep_count = dnode->data_.extern_rc;
      for(DagNode* succ : node->successors_) {
        NodeState succ_state = node_states_.GetState(succ->node_id());
        if(succ_state == NodeState::kBirth || succ_state == NodeState::kReady) {
          ++dep_count;
        }
      }
      if(dep_count == 0) {
        node_states_.ChangeState(nid, NodeState::kDead);
      }
      DataStore& ds = MinervaSystem::Instance().data_store();
      if(ds.ExistData(dnode->data_.data_id)) {
        ds.SetReferenceCount(dnode->data_.data_id, dep_count);
      }
      break;
    }
  }
  for(uint64_t nid : node_states_.GetNodesOfState(NodeState::kDead)) {
    dag.DeleteNode(nid);
  }
}

void PhysicalEngine::NodeRunner(DagNode* node) {
  MinervaSystem& ms = MinervaSystem::Instance();
  uint64_t nid = node->node_id();
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
    LOG(INFO) << "Execute node#" << nid << " compute fn: " << op.compute_fn->Name();
    op.compute_fn->Execute(input, output, BASIC); // TODO decide impl_type
    for (auto n: phy_op_node->predecessors_) {// de-refer predecessor's data
      PhysicalData& in_data = dynamic_cast<PhysicalDataNode*>(n)->data_;
      ms.data_store().DecrReferenceCount(in_data.data_id);
    }
  } 
  // trigger successors
  {
    lock_guard<mutex> lock(node_states_mutex_);
    for (auto succ: node->successors_) {
      NodeState state = node_states_.GetState(succ->node_id());
      RuntimeState& rts = rt_states_[succ->node_id()];
      CHECK_GE(--rts.dependency_counter, 0) << "";
      // Append node if all predecessors are finished
      if (state == NodeState::kReady && rts.dependency_counter == 0) {
        AppendTask(succ, bind(&PhysicalEngine::NodeRunner, this, placeholders::_1));
      }
    }
    node_states_.ChangeState(nid, NodeState::kCompleted);
    if (rt_states_[nid].on_complete != nullptr) {
      //printf("Target complete %u\n", (unsigned int) dynamic_cast<PhysicalDataNode*>(node)->data_.data_id);
      rt_states_[nid].on_complete->notify_all();
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
