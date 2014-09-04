#include "physical_engine.h"
#include "system/minerva_system.h"
#include <gflags/gflags.h>
#include <iostream>

using namespace std;

namespace minerva {
  
PhysicalEngine::PhysicalEngine(ThreadPool& tp, DataStore& ds): DagEngine<PhysicalDag>(tp), data_store_(ds) {
}
  
void PhysicalEngine::SetUpReadyNodeState(DagNode* node) {
  impl_decider_->Decide(node, node_states_);
}
  
void PhysicalEngine::FreeDataNodeRes(PhysicalDataNode* dnode) {
  data_store_.SetReferenceCount(dnode->data_.data_id, 0);
}

std::unordered_set<uint64_t> PhysicalEngine::FindStartFrontier(PhysicalDag& dag, const std::vector<uint64_t>& targets) {
  std::unordered_set<uint64_t> start_frontier;
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
      start_frontier.insert(nid);
    }
  }
  return start_frontier;
}

void PhysicalEngine::ProcessNode(DagNode* node) {
  uint64_t nid = node->node_id();
  if (node->Type() == DagNode::OP_NODE) { // OpNode
/*
    vector<DataShard> input;
    vector<DataShard> output;
    for (auto n: phy_op_node->inputs_) {
      PhysicalData& in_data = dynamic_cast<PhysicalDataNode*>(n)->data_;
      input.push_back(DataShard(in_data));
    }
    for (auto n: phy_op_node->outputs_) { // Allocate storage for all outputs
      PhysicalData& out_data = dynamic_cast<PhysicalDataNode*>(n)->data_;
      data_store_.CreateData(out_data.data_id, DataStore::CPU, out_data.size.Prod(), 1);
      output.push_back(DataShard(out_data));
    }
*/
    // call compute function
    PhysicalOpNode* phy_op_node = dynamic_cast<PhysicalOpNode*>(node);
    PhysicalOp& op = phy_op_node->op_;
    uint64_t device_id = op.compute_fn->device_info.id;
    Device* device = MinervaSystem::Instance().GetDevice(device_id);
    CHECK_NOTNULL(device);
    vector<PhysicalData> inputs;
    for (auto n: phy_op_node->inputs_)
      inputs.push_back(dynamic_cast<PhysicalDataNode*>(n)->data_);
    vector<PhysicalData> outputs;
    for (auto n: phy_op_node->outputs_)
      outputs.push_back(dynamic_cast<PhysicalDataNode*>(n)->data_);
    device->Execute(nid, inputs, outputs, op);
  }
}

}
