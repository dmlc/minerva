#include <random>
#include <queue>
#include <glog/logging.h>
#include <gflags/gflags.h>

#include "procedures/physical_engine.h"
#include "op/physical.h"
#include "system/minerva_system.h"

using namespace std;

/////////////////////// flag definitions //////////////////////
static bool IsValidNumThreads(const char* flag, int n) {
  return n > 0;
}
DEFINE_int32(numthreads, 2, "number of threads used in execution");
static const bool numthreads_valid = gflags::RegisterFlagValidator(&FLAGS_numthreads, &IsValidNumThreads);
DEFINE_bool(enable_execute, true, "enable concrete computation");

/////////////////////// member function definitions //////////////////////
namespace minerva {

PhysicalEngine::PhysicalEngine(): thread_pool_(FLAGS_numthreads) {
  Init();
}

PhysicalEngine::~PhysicalEngine() {
}

void PhysicalEngine::Process(PhysicalDag& dag, NodeStateMap<PhysicalDag>& node_states,
    const std::vector<uint64_t>& targets) {
  auto ready_to_execute = FindRootNodes(dag, node_states, targets);
  for (auto ready_node: ready_to_execute) {
    AppendTask(ready_node, node_states);
  }
  {
    // Waiting execution to complete
    for (uint64_t tgtid: targets) {
      unique_lock<mutex> lock(*rt_states_[tgtid].state_mutex);
      LOG(INFO) << "Wait for node (id=" << tgtid << ") finish.";
      if (node_states.GetState(tgtid) != NodeState::kCompleted) {
        rt_states_[tgtid].on_complete->wait(lock);
      }
      delete rt_states_[tgtid].on_complete;
      rt_states_[tgtid].on_complete = nullptr;
      LOG(INFO) << "Node (id=" << tgtid << ") complete.";
    }
  }
  thread_pool_.WaitForAllFinished();
}

void PhysicalEngine::Init() {
  // Then we can load user defined runners
}

void PhysicalEngine::OnCreateNode(DagNode* node) {
  RuntimeState ns{0, NULL, new std::mutex()};
  rt_states_.insert(make_pair(node->node_id(), ns));
}

void PhysicalEngine::OnDeleteNode(DagNode* node) {
  delete rt_states_[node->node_id()].state_mutex;
  rt_states_.erase(node->node_id());
}

unordered_set<DagNode*> PhysicalEngine::FindRootNodes(PhysicalDag& dag, NodeStateMap<PhysicalDag>& node_states,
    const vector<uint64_t>& targets) {
  queue<uint64_t> ready_node_queue;
  unordered_set<DagNode*> ready_to_execute;
  for (uint64_t tgtid: targets) {
    // Don't push complete nodes
    if (node_states.GetState(tgtid) != NodeState::kCompleted) {
      ready_node_queue.push(tgtid);
    }
  }
  while (!ready_node_queue.empty()) {
    uint64_t curid = ready_node_queue.front();
    ready_node_queue.pop();
    node_states.ChangeState(curid, NodeState::kReady);
    RuntimeState& rts = rt_states_[curid];
    auto node = dag.GetNode(curid);;
    rts.dependency_counter = 0;
    for (auto pred: node->predecessors_) {
      switch (node_states.GetState(pred->node_id())) {
        // Count dependency and recursively search predecessors
        case NodeState::kBirth:
          ready_node_queue.push(pred->node_id());
          ++rts.dependency_counter;
          break;
        case NodeState::kReady:
          ++rts.dependency_counter;
          break;
        default:
          break;
      }
    }
    // All successors of OpNode will be set ready. Successor of an incomplete OpNode could not be complete.
    if (node->Type() == DagNode::OP_NODE) {
      for (auto succ: node->successors_) {
        node_states.ChangeState(succ->node_id(), NodeState::kReady);
      }
    }
    // All predecessors are complete, or there are no predecessors at all
    if (rts.dependency_counter == 0) {
      ready_to_execute.insert(node);
    }
  }
  for (uint64_t tgtid: targets) {
    if (node_states.GetState(tgtid) != NodeState::kCompleted) {
      rt_states_[tgtid].on_complete = new condition_variable;
    }
  }
  return ready_to_execute;
}

void PhysicalEngine::GCNodes(PhysicalDag& dag, NodeStateMap<PhysicalDag>& node_states) {
  vector<uint64_t> dead_nodes;
  for(uint64_t nid : node_states.GetNodesOfState(NodeState::kDead)) {
    dead_nodes.push_back(nid);
  }
  //cout << "#completed/#dead/: " << node_states.GetNodesOfState(NodeState::kCompleted).size() 
    //<< "/" << dead_nodes.size() << endl;
  for(uint64_t nid : dead_nodes) {
    dag.DeleteNode(nid);
  }
}

void PhysicalEngine::AppendTask(DagNode* node, NodeStateMap<PhysicalDag>& node_states) {
  thread_pool_.Push(bind(&PhysicalEngine::NodeRunner, this, node, std::ref(node_states)));
}

void PhysicalEngine::NodeRunner(DagNode* node, NodeStateMap<PhysicalDag>& node_states) {
  MinervaSystem& ms = MinervaSystem::Instance();
  uint64_t nid = node->node_id();
  CHECK_EQ(node_states.GetState(nid), NodeState::kReady);
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
    if(FLAGS_enable_execute) {
      DLOG(INFO) << "Execute node#" << nid << " compute fn: " << op.compute_fn->Name();
      op.compute_fn->Execute(input, output, op.impl_type);
    }
    for (auto n: phy_op_node->predecessors_) {// de-refer predecessor's data
      PhysicalDataNode* pred_dnode = dynamic_cast<PhysicalDataNode*>(n);
      PhysicalData& in_data = pred_dnode->data_;
      ms.data_store().DecrReferenceCount(in_data.data_id);
      if(in_data.extern_rc == 0) {
        node_states.ChangeState(pred_dnode->node_id(), NodeState::kDead);
      }
    }
  } 
  {
    //lock_guard<mutex> lock(*rt_states_[nid].state_mutex);
    //// ATTENTION: we don't need this lock if we could assure that for each required nid,
    //              the NodeRunner() function would be called once and only once.
    // change states
    if(node->Type() == DagNode::OP_NODE) {
      node_states.ChangeState(nid, NodeState::kDead); // the op node is executed thus could be GCed
    } else {
      PhysicalDataNode* pdnode = dynamic_cast<PhysicalDataNode*>(node);
      if(pdnode->data_.extern_rc != 0) {
        // the data node is completed but with external dependencies
        node_states.ChangeState(nid, NodeState::kCompleted);
      } else {
        // the data node could be GCed
        node_states.ChangeState(nid, NodeState::kDead);
      }
    }
    // trigger on_complete hook
    if (rt_states_[nid].on_complete != nullptr) {
      //printf("Target complete %u\n", (unsigned int) dynamic_cast<PhysicalDataNode*>(node)->data_.data_id);
      rt_states_[nid].on_complete->notify_all();
    }
  }
  // trigger successors
  for (auto succ: node->successors_) {
    RuntimeState& rts = rt_states_[succ->node_id()];
    lock_guard<mutex> lock(*rts.state_mutex);
    CHECK_GE(--rts.dependency_counter, 0) << "wrong dependency_counter for node#" << succ->node_id();
    // Append node if all predecessors are finished
    NodeState state = node_states.GetState(succ->node_id());
    if (state == NodeState::kReady && rts.dependency_counter == 0) {
      AppendTask(succ, node_states);
    }
  }
}

} // end of namespace minerva
