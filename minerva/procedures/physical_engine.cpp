#include "physical_engine.h"
#include <gflags/gflags.h>

DEFINE_bool(enable_execute, true, "enable concrete computation");

using namespace std;

namespace minerva {
  
PhysicalEngine::PhysicalEngine(ThreadPool& tp, DataStore& ds): DagEngine<PhysicalDag>(tp), data_store_(ds) {
}
  
void PhysicalEngine::FreeNodeResources(PhysicalDataNode* dnode) {
  data_store_.SetReferenceCount(dnode->data_.data_id, 0);
}

void PhysicalEngine::ProcessNode(DagNode* node) {
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
      data_store_.CreateData(out_data.data_id, DataStore::CPU, out_data.size.Prod(), 1);
      output.push_back(DataShard(out_data));
    }
    // call compute function
    PhysicalOp& op = phy_op_node->op_;
    CHECK_NOTNULL(op.compute_fn);
    if(FLAGS_enable_execute) {
      DLOG(INFO) << "Execute node#" << nid << " compute fn: " << op.compute_fn->Name();
      op.compute_fn->Execute(input, output, op.impl_type);
    }
  }
}

}
